"""
lazyRLGYM — LoRA Adapter Merging
=================================
Utilities for merging LoRA adapters into base model weights and loading
merged models for inference.

After SFT or DPO training, the LoRA adapter is saved separately. For inference,
we merge the adapter weights into the base model to create a standalone
checkpoint that doesn't need peft at runtime. This is important because:
  1. Inference doesn't need LoRA's adapter overhead
  2. The merged model can be loaded with a single from_pretrained call
  3. It simplifies the inference pipeline (no PeftModel wrapping)

Memory for merging 1.5B model:
  - Base model in bf16:  ~3 GB
  - Merged model in bf16: ~3 GB (same, weights are overwritten in-place)
  - Total:               ~3-6 GB (fits easily in 12 GB)
"""
from __future__ import annotations

import gc
import logging
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import MODEL_DTYPE
from utils.vram import get_vram_usage

logger = logging.getLogger(__name__)


def merge_and_save(base_model_path: str, adapter_path: str, output_path: str) -> None:
    """Merge a LoRA adapter into the base model and save the merged checkpoint.

    Args:
        base_model_path: HuggingFace model ID or local path to the base model.
        adapter_path: Path to the saved LoRA adapter directory.
        output_path: Directory to save the merged model.

    Raises:
        FileNotFoundError: If the adapter path does not exist.
        RuntimeError: If merging fails.
    """
    from pathlib import Path

    adapter_path_obj = Path(adapter_path)
    if not adapter_path_obj.exists():
        raise FileNotFoundError(
            f"LoRA adapter not found at {adapter_path}. "
            "Ensure training has saved the adapter to this location."
        )

    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    # Determine torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    _dtype = dtype_map.get(MODEL_DTYPE, torch.bfloat16)

    logger.info(f"[Merge] Loading base model from {base_model_path} (dtype={MODEL_DTYPE})")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        dtype=_dtype,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    )

    logger.info(f"[Merge] Loading LoRA adapter from {adapter_path}")
    peft_model = PeftModel.from_pretrained(base_model, adapter_path)

    logger.info("[Merge] Merging adapter weights into base model")
    merged_model = peft_model.merge_and_unload()

    # Save merged model
    logger.info(f"[Merge] Saving merged model to {output_path}")
    merged_model.save_pretrained(str(output_path_obj), max_shard_size="2GB")

    # Save tokenizer alongside the merged model
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(str(output_path_obj))

    logger.info(f"[Merge] Merge complete. VRAM: {get_vram_usage()}")

    # Cleanup
    del merged_model
    del peft_model
    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    logger.info(f"[Merge] Cleanup done. VRAM: {get_vram_usage()}")


def load_for_inference(merged_path: str) -> tuple[Any, Any]:
    """Load a merged model for inference.

    Args:
        merged_path: Path to the merged model directory.

    Returns:
        A tuple of (model, tokenizer) ready for generation.

    Raises:
        FileNotFoundError: If the merged model path does not exist.
    """
    from pathlib import Path

    merged_path_obj = Path(merged_path)
    if not merged_path_obj.exists():
        raise FileNotFoundError(
            f"Merged model not found at {merged_path}. "
            "Run merge_and_save() first to create the merged checkpoint."
        )

    # Determine torch dtype
    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    _dtype = dtype_map.get(MODEL_DTYPE, torch.bfloat16)

    logger.info(f"[Inference] Loading merged model from {merged_path} (dtype={MODEL_DTYPE})")
    model = AutoModelForCausalLM.from_pretrained(
        merged_path,
        dtype=_dtype,
        attn_implementation="sdpa",
    )

    # Enable cache for inference (faster generation)
    model.config.use_cache = True
    model.eval()

    logger.info(f"[Inference] Loading tokenizer from {merged_path}")
    tokenizer = AutoTokenizer.from_pretrained(merged_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info(f"[Inference] Model loaded. VRAM: {get_vram_usage()}")

    return model, tokenizer
