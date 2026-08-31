"""
lazyRLGYM — SFT Warmup Training
================================
Supervised fine-tuning warmup phase on Fable-5 traces using LoRA adapters.

Trains DeepSeek-R1-Distill-Qwen-1.5B with LoRA (peft 0.15.0) via TRL 0.24.0's
SFTTrainer, designed to fit within 12GB VRAM on an RTX 3060.

Memory budget (bf16):
  - 1.5B base model weights:  ~3 GB
  - LoRA r=32 adapters:       ~0.1 GB
  - Optimizer states (8-bit): ~0.1 GB
  - Activations (grad ckpt):  ~2-4 GB
  - Total:                    ~6-8 GB  (fits in 12 GB)
"""
from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer as TRLSFTTrainer

from config import TrainConfig, MODEL_DTYPE, MAX_SEQ_LEN
from utils.vram import register, unload, get_vram_usage

logger = logging.getLogger(__name__)

# Alias to avoid clashing with our wrapper class name
_TRLSFTTrainer = TRLSFTTrainer


class SFTTrainer:
    """SFT warmup trainer with LoRA adapters for the lazyRLGYM pipeline.

    Wraps TRL 0.24.0's SFTTrainer, handling LoRA setup, gradient checkpointing,
    VRAM registration, and OOM recovery on a 12GB GPU.
    """

    def __init__(self, model_path: str, cfg: TrainConfig):
        """Load the base model and prepare LoRA configuration.

        Args:
            model_path: HuggingFace model ID or local path to the base model.
            cfg: TrainConfig containing LoRA and SFT hyperparameters.
        """
        self.cfg = cfg
        self.model_path = model_path
        self.model: Any = None
        self.tokenizer: Any = None
        self.peft_config: LoraConfig | None = None
        self.trainer: _TRLSFTTrainer | None = None

        # Determine torch dtype from config string
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(MODEL_DTYPE, torch.bfloat16)

        # Build LoRA config
        self.peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Load tokenizer
        logger.info(f"[SFT] Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("[SFT] Tokenizer had no pad_token; set to eos_token")

        # Load base model in bf16
        logger.info(f"[SFT] Loading base model from {model_path} (dtype={MODEL_DTYPE})")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=self.torch_dtype,
            attn_implementation="sdpa",  # compatible with Windows / no flash-attn
        )
        self.model.config.use_cache = False  # required for gradient checkpointing

        # Enable gradient checkpointing before wrapping with PEFT
        if cfg.gradient_checkpointing:
            logger.info("[SFT] Enabling gradient checkpointing")
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

        # Register with VRAM tracker
        register("sft_model", self.model)
        logger.info(f"[SFT] Model loaded. VRAM: {get_vram_usage()}")

    def _build_sft_config(self, output_dir: Path) -> SFTConfig:
        """Construct SFTConfig from our TrainConfig."""
        return SFTConfig(
            output_dir=str(output_dir),
            num_train_epochs=self.cfg.sft_epochs,
            per_device_train_batch_size=self.cfg.sft_batch_size,
            gradient_accumulation_steps=self.cfg.sft_grad_accum,
            learning_rate=self.cfg.sft_lr,
            lr_scheduler_type="cosine",
            warmup_ratio=0.05,
            logging_steps=5,
            save_strategy="no",
            save_total_limit=self.cfg.save_total_limit,
            bf16=True,
            gradient_checkpointing=self.cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=self.cfg.dataloader_num_workers,
            group_by_length=self.cfg.group_by_length,
            max_length=min(MAX_SEQ_LEN, 2048),  # cap for VRAM safety
            packing=False,
            assistant_only_loss=True,  # only compute loss on assistant tokens
            optim="paged_adamw_8bit",
            remove_unused_columns=False,
        )

    def train(self, dataset: list[dict], output_dir: Path) -> dict:
        """Run SFT training on the provided conversational dataset.

        Args:
            dataset: List of {"messages": [{"role": ..., "content": ...}, ...]}.
            output_dir: Directory to save the LoRA adapter.

        Returns:
            {"loss": float, "output_dir": str}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not dataset:
            raise ValueError("[SFT] Empty dataset provided to train()")

        logger.info(f"[SFT] Preparing dataset with {len(dataset)} examples")

        # Convert list[dict] to HuggingFace Dataset
        # Each example: {"messages": [{"role": "user", "content": ...}, ...]}
        hf_dataset = Dataset.from_list(dataset)

        # Build SFT config
        sft_config = self._build_sft_config(output_dir)

        # Attempt training with OOM recovery
        return self._train_with_oom_recovery(hf_dataset, sft_config, output_dir)

    def _train_with_oom_recovery(
        self, hf_dataset: Dataset, sft_config: SFTConfig, output_dir: Path
    ) -> dict:
        """Run training, reducing batch size / switching to 8-bit optim on OOM.

        Recovery strategy (in order):
          1. Default: adamw_torch, configured batch size
          2. OOM: halve batch_size, double grad_accum, switch to paged_adamw_8bit
          3. OOM again: batch_size=1, grad_accum=original*4, paged_adamw_8bit
        """
        max_retries = 3
        current_config = sft_config

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"[SFT] Training attempt {attempt + 1}/{max_retries} "
                    f"(batch_size={current_config.per_device_train_batch_size}, "
                    f"grad_accum={current_config.gradient_accumulation_steps}, "
                    f"optim={current_config.optim})"
                )

                # Create TRL SFTTrainer — it handles peft wrapping internally
                # when peft_config is provided.
                self.trainer = _TRLSFTTrainer(
                    model=self.model,
                    args=current_config,
                    train_dataset=hf_dataset,
                    processing_class=self.tokenizer,
                    peft_config=self.peft_config,
                )

                # Train
                train_result = self.trainer.train()

                # Save the LoRA adapter
                logger.info(f"[SFT] Saving LoRA adapter to {output_dir}")
                self.trainer.save_model(str(output_dir))
                self.tokenizer.save_pretrained(str(output_dir))

                loss = train_result.training_loss
                logger.info(f"[SFT] Training complete. Loss={loss:.4f}")
                logger.info(f"[SFT] VRAM after training: {get_vram_usage()}")

                return {"loss": float(loss), "output_dir": str(output_dir)}

            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"[SFT] OOM on attempt {attempt + 1}: {e}")
                self._cleanup_trainer()
                current_config = self._reduce_config(current_config, attempt)
                if current_config is None:
                    logger.error("[SFT] Exhausted OOM recovery options")
                    raise

            except RuntimeError as e:
                # Catch CUDA errors that don't use the OOM subclass
                if "out of memory" in str(e).lower():
                    logger.warning(f"[SFT] CUDA OOM on attempt {attempt + 1}: {e}")
                    self._cleanup_trainer()
                    current_config = self._reduce_config(current_config, attempt)
                    if current_config is None:
                        logger.error("[SFT] Exhausted OOM recovery options")
                        raise
                else:
                    logger.error(f"[SFT] Runtime error: {e}")
                    raise

        # Should not reach here, but just in case
        raise RuntimeError("[SFT] Training failed after all OOM recovery attempts")

    def _reduce_config(self, config: SFTConfig, attempt: int) -> SFTConfig | None:
        """Reduce memory usage for the next training attempt.

        Args:
            config: Current SFTConfig.
            attempt: Zero-based attempt index.

        Returns:
            A new SFTConfig with reduced memory settings, or None if no
            further reduction is possible.
        """
        if attempt == 0:
            # Halve batch size, double grad accum, switch to 8-bit optim
            new_bs = max(1, config.per_device_train_batch_size // 2)
            new_ga = config.gradient_accumulation_steps * 2
            logger.info(
                f"[SFT] Reducing: batch_size {config.per_device_train_batch_size}->{new_bs}, "
                f"grad_accum {config.gradient_accumulation_steps}->{new_ga}, "
                f"optim->paged_adamw_8bit"
            )
            config.per_device_train_batch_size = new_bs
            config.gradient_accumulation_steps = new_ga
            config.optim = "paged_adamw_8bit"
            return config

        elif attempt == 1:
            # Minimize batch size, maximize grad accum
            logger.info("[SFT] Reducing: batch_size->1, grad_accum->4x original")
            config.per_device_train_batch_size = 1
            config.gradient_accumulation_steps = self.cfg.sft_grad_accum * 4
            config.max_length = min(config.max_length or 4096, 2048)
            return config

        else:
            return None

    def _cleanup_trainer(self):
        """Clean up the trainer object and free VRAM between OOM retries."""
        if self.trainer is not None:
            # Force delete optimizer/lr_scheduler references
            try:
                if hasattr(self.trainer, "optimizer"):
                    self.trainer.optimizer = None
                if hasattr(self.trainer, "lr_scheduler"):
                    self.trainer.lr_scheduler = None
            except Exception:
                pass
            del self.trainer
            self.trainer = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info(f"[SFT] After OOM cleanup: {get_vram_usage()}")

    def unload(self):
        """Free all VRAM held by the model and trainer."""
        logger.info("[SFT] Unloading model and freeing VRAM")
        self._cleanup_trainer()

        if self.model is not None:
            unload("sft_model")
            self.model = None

        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info(f"[SFT] After unload: {get_vram_usage()}")
