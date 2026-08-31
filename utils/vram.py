"""
VRAM management — load/unload models to fit in 12GB.
The 1.5B model can't do inference AND training simultaneously.
This module provides a context manager that tracks what's loaded
and enforces cleanup between phases.
"""
import gc
import torch
from typing import Optional, Any


_loaded_models: dict[str, Any] = {}


def get_vram_usage() -> dict:
    """Return current VRAM usage in MB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "free": 0, "total": 0}
    allocated = torch.cuda.memory_allocated() / 1024 / 1024
    reserved = torch.cuda.memory_reserved() / 1024 / 1024
    total = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
    free = total - allocated
    return {"allocated": allocated, "reserved": reserved, "free": free, "total": total}


def unload_all():
    """Unload all tracked models and free VRAM."""
    global _loaded_models
    for name in list(_loaded_models.keys()):
        model = _loaded_models.pop(name)
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    print(f"[VRAM] After cleanup: {get_vram_usage()}")


def unload(name: str):
    """Unload a specific named model."""
    global _loaded_models
    if name in _loaded_models:
        model = _loaded_models.pop(name)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[VRAM] Unloaded '{name}'. Now: {get_vram_usage()}")


def register(name: str, model: Any):
    """Register a model so it can be tracked and unloaded."""
    _loaded_models[name] = model
    print(f"[VRAM] Loaded '{name}'. Now: {get_vram_usage()}")


def is_loaded(name: str) -> bool:
    return name in _loaded_models


def get_model(name: str) -> Optional[Any]:
    return _loaded_models.get(name)
