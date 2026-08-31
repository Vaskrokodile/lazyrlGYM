"""
lazyRLGYM — DPO Training with LoRA
===================================
Iterative DPO training phase using length-difficulty-shaped preference pairs.

Trains DeepSeek-R1-Distill-Qwen-1.5B with LoRA (peft 0.15.0) via TRL 0.24.0's
DPOTrainer, designed to fit within 12GB VRAM on an RTX 3060.

Memory budget (bf16, LoRA with adapter-disabled ref):
  - 1.5B base model weights (shared policy+ref):  ~3 GB
  - LoRA r=32 adapters:                           ~0.1 GB
  - Optimizer states (8-bit):                     ~0.1 GB
  - Activations (grad ckpt, 2 forward passes):    ~4-6 GB
  - Total:                                        ~8-10 GB (fits in 12 GB)

Key insight: TRL 0.24.0's DPOTrainer, when given a peft_config and ref_model=None,
uses the base model with LoRA adapters *disabled* as the reference model. This
avoids loading a separate 3GB copy of the base model for the reference, which is
critical for fitting DPO in 12GB VRAM.
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
from trl import DPOConfig, DPOTrainer as TRLDPOTrainer

from config import TrainConfig, MODEL_DTYPE, MAX_SEQ_LEN
from utils.vram import register, unload, get_vram_usage

logger = logging.getLogger(__name__)

# Alias to avoid clashing with our wrapper class name
_TRLDPOTrainer = TRLDPOTrainer


class DPOTrainer:
    """DPO trainer with LoRA adapters for the lazyRLGYM pipeline.

    Wraps TRL 0.24.0's DPOTrainer. When using LoRA, the reference model is the
    base model with adapters disabled (handled internally by TRL), avoiding the
    need for a separate ref model copy in VRAM.
    """

    def __init__(self, model_path: str, cfg: TrainConfig):
        """Load the base model and prepare LoRA configuration for DPO.

        Args:
            model_path: HuggingFace model ID or local path to the base model.
            cfg: TrainConfig containing LoRA and DPO hyperparameters.
        """
        self.cfg = cfg
        self.model_path = model_path
        self.model: Any = None
        self.tokenizer: Any = None
        self.peft_config: LoraConfig | None = None
        self.trainer: _TRLDPOTrainer | None = None

        # Determine torch dtype from config string
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        self.torch_dtype = dtype_map.get(MODEL_DTYPE, torch.bfloat16)

        # Build LoRA config — same as SFT, applied to the policy model
        self.peft_config = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

        # Load tokenizer
        logger.info(f"[DPO] Loading tokenizer from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("[DPO] Tokenizer had no pad_token; set to eos_token")

        # Load base model in bf16
        # This single model serves as both the policy (with LoRA adapters enabled)
        # and the reference (with LoRA adapters disabled). TRL handles this
        # internally when peft_config is provided and ref_model=None.
        logger.info(f"[DPO] Loading base model from {model_path} (dtype={MODEL_DTYPE})")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=self.torch_dtype,
            attn_implementation="sdpa",  # compatible with Windows / no flash-attn
        )
        self.model.config.use_cache = False  # required for gradient checkpointing

        # Enable gradient checkpointing before TRL wraps with PEFT
        if cfg.gradient_checkpointing:
            logger.info("[DPO] Enabling gradient checkpointing")
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            if hasattr(self.model, "enable_input_require_grads"):
                self.model.enable_input_require_grads()

        # Register with VRAM tracker
        register("dpo_model", self.model)
        logger.info(f"[DPO] Model loaded. VRAM: {get_vram_usage()}")

    def _build_dpo_config(self, output_dir: Path) -> DPOConfig:
        """Construct DPOConfig from our TrainConfig."""
        return DPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=self.cfg.dpo_epochs,
            per_device_train_batch_size=self.cfg.dpo_batch_size,
            gradient_accumulation_steps=self.cfg.dpo_grad_accum,
            learning_rate=self.cfg.dpo_lr,
            lr_scheduler_type="cosine",
            warmup_ratio=self.cfg.dpo_warmup_ratio,
            logging_steps=5,
            save_strategy="no",
            save_total_limit=self.cfg.save_total_limit,
            bf16=True,
            gradient_checkpointing=self.cfg.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=self.cfg.dataloader_num_workers,
            group_by_length=False,  # DPO dataset has no input_ids; can't group by length
            # DPO-specific
            beta=self.cfg.dpo_beta,
            max_length=min(self.cfg.dpo_max_length, MAX_SEQ_LEN),
            max_prompt_length=min(self.cfg.dpo_max_length // 2, 2048),
            loss_type="sigmoid",
            # Memory: precompute ref log probs to avoid keeping ref in VRAM
            # during training. With LoRA, ref is the adapter-disabled base model,
            # so this is optional but can help if activations are the bottleneck.
            precompute_ref_log_probs=False,
            optim="paged_adamw_8bit",
            remove_unused_columns=False,
        )

    def train(self, preference_pairs: list[dict], output_dir: Path) -> dict:
        """Run DPO training on the provided preference pairs.

        Args:
            preference_pairs: List of {"prompt": str, "chosen": str, "rejected": str}.
            output_dir: Directory to save the LoRA adapter.

        Returns:
            {"loss": float, "reward_accuracy": float, "output_dir": str}
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if not preference_pairs:
            raise ValueError("[DPO] Empty preference_pairs provided to train()")

        logger.info(f"[DPO] Preparing dataset with {len(preference_pairs)} pairs")

        # Convert list[dict] to HuggingFace Dataset
        # Each example: {"prompt": str, "chosen": str, "rejected": str}
        hf_dataset = Dataset.from_list(preference_pairs)

        # Build DPO config
        dpo_config = self._build_dpo_config(output_dir)

        # Attempt training with OOM recovery
        return self._train_with_oom_recovery(hf_dataset, dpo_config, output_dir)

    def _train_with_oom_recovery(
        self, hf_dataset: Dataset, dpo_config: DPOConfig, output_dir: Path
    ) -> dict:
        """Run training, reducing batch size / switching to 8-bit optim on OOM.

        Recovery strategy (in order):
          1. Default: adamw_torch, configured batch size
          2. OOM: halve batch_size, double grad_accum, switch to paged_adamw_8bit
          3. OOM again: batch_size=1, grad_accum=original*4, paged_adamw_8bit,
             reduce max_length
        """
        max_retries = 3
        current_config = dpo_config

        for attempt in range(max_retries):
            try:
                logger.info(
                    f"[DPO] Training attempt {attempt + 1}/{max_retries} "
                    f"(batch_size={current_config.per_device_train_batch_size}, "
                    f"grad_accum={current_config.gradient_accumulation_steps}, "
                    f"optim={current_config.optim}, "
                    f"max_length={current_config.max_length})"
                )

                # Create TRL DPOTrainer.
                # When peft_config is provided and ref_model=None, TRL uses the
                # base model with adapters disabled as the reference model.
                # This is the memory-efficient path for 12GB VRAM.
                self.trainer = _TRLDPOTrainer(
                    model=self.model,
                    ref_model=None,  # TRL uses adapter-disabled base as ref
                    args=current_config,
                    train_dataset=hf_dataset,
                    processing_class=self.tokenizer,
                    peft_config=self.peft_config,
                )

                # Train
                train_result = self.trainer.train()

                # Save the LoRA adapter
                logger.info(f"[DPO] Saving LoRA adapter to {output_dir}")
                self.trainer.save_model(str(output_dir))
                self.tokenizer.save_pretrained(str(output_dir))

                # Extract metrics from training log history
                loss = train_result.training_loss
                reward_accuracy = self._extract_reward_accuracy()

                logger.info(
                    f"[DPO] Training complete. Loss={loss:.4f}, "
                    f"reward_accuracy={reward_accuracy:.4f}"
                )
                logger.info(f"[DPO] VRAM after training: {get_vram_usage()}")

                return {
                    "loss": float(loss),
                    "reward_accuracy": float(reward_accuracy),
                    "output_dir": str(output_dir),
                }

            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"[DPO] OOM on attempt {attempt + 1}: {e}")
                self._cleanup_trainer()
                current_config = self._reduce_config(current_config, attempt)
                if current_config is None:
                    logger.error("[DPO] Exhausted OOM recovery options")
                    raise

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"[DPO] CUDA OOM on attempt {attempt + 1}: {e}")
                    self._cleanup_trainer()
                    current_config = self._reduce_config(current_config, attempt)
                    if current_config is None:
                        logger.error("[DPO] Exhausted OOM recovery options")
                        raise
                else:
                    logger.error(f"[DPO] Runtime error: {e}")
                    raise

        raise RuntimeError("[DPO] Training failed after all OOM recovery attempts")

    def _extract_reward_accuracy(self) -> float:
        """Extract the final reward accuracy from the trainer's log history.

        TRL's DPOTrainer logs 'rewards/accuracies' during training. We pull the
        last logged value.
        """
        if self.trainer is None or not hasattr(self.trainer, "state"):
            return 0.0

        log_history = self.trainer.state.log_history
        if not log_history:
            return 0.0

        # Search from the end for the most recent rewards/accuracies entry
        for entry in reversed(log_history):
            if "rewards/accuracies" in entry:
                return float(entry["rewards/accuracies"])

        return 0.0

    def _reduce_config(self, config: DPOConfig, attempt: int) -> DPOConfig | None:
        """Reduce memory usage for the next training attempt.

        Args:
            config: Current DPOConfig.
            attempt: Zero-based attempt index.

        Returns:
            A new DPOConfig with reduced memory settings, or None if no
            further reduction is possible.
        """
        if attempt == 0:
            # Halve batch size, double grad accum, switch to 8-bit optim
            new_bs = max(1, config.per_device_train_batch_size // 2)
            new_ga = config.gradient_accumulation_steps * 2
            logger.info(
                f"[DPO] Reducing: batch_size {config.per_device_train_batch_size}->{new_bs}, "
                f"grad_accum {config.gradient_accumulation_steps}->{new_ga}, "
                f"optim->paged_adamw_8bit"
            )
            config.per_device_train_batch_size = new_bs
            config.gradient_accumulation_steps = new_ga
            config.optim = "paged_adamw_8bit"
            return config

        elif attempt == 1:
            # Minimize batch size, maximize grad accum, reduce max_length
            logger.info(
                "[DPO] Reducing: batch_size->1, grad_accum->4x original, "
                "max_length->2048"
            )
            config.per_device_train_batch_size = 1
            config.gradient_accumulation_steps = self.cfg.dpo_grad_accum * 4
            config.max_length = min(config.max_length or 4096, 2048)
            config.max_prompt_length = min(config.max_prompt_length or 1024, 1024)
            return config

        else:
            return None

    def _cleanup_trainer(self):
        """Clean up the trainer object and free VRAM between OOM retries."""
        if self.trainer is not None:
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
        logger.info(f"[DPO] After OOM cleanup: {get_vram_usage()}")

    def unload(self):
        """Free all VRAM held by the model and trainer.

        With LoRA + ref_model=None, there is only one model in VRAM (the base
        model with adapters). The reference model was never separately loaded.
        """
        logger.info("[DPO] Unloading model and freeing VRAM")
        self._cleanup_trainer()

        if self.model is not None:
            unload("dpo_model")
            self.model = None

        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.info(f"[DPO] After unload: {get_vram_usage()}")
