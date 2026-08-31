"""
lazyRLGYM — Configuration
=========================
Central config for the entire RL training pipeline.
All modules import from here.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(r"E:\lazyRLGYM")
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
LOG_DIR = PROJECT_ROOT / "logs"
DATA_CACHE_DIR = PROJECT_ROOT / "data" / "cache"

for d in [CHECKPOINT_DIR, LOG_DIR, DATA_CACHE_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── Model ──────────────────────────────────────────────────────────────────
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
MODEL_DTYPE = "bfloat16"          # bf16 fits in 12GB for 1.5B
MAX_SEQ_LEN = 8192                # max context for rollouts
THINK_TOKEN = "<think>"           # R1 thinking block start
END_THINK_TOKEN = "</think>"      # R1 thinking block end


# ── Hardware ───────────────────────────────────────────────────────────────
GPU_VRAM_GB = 12                  # RTX 3060 12GB
RAM_GB = 16
DEVICE = "cuda"


# ── Rollout ────────────────────────────────────────────────────────────────
@dataclass
class RolloutConfig:
    n_samples_per_prompt: int = 6         # N completions per prompt (more = more pairs)
    max_new_tokens: int = 1024            # max generation length
    temperature: float = 0.8              # slightly higher for more diversity
    top_p: float = 0.95
    batch_size: int = 6                   # prompts per generate call (batched)
    n_difficulty_samples: int = 0         # unused — difficulty from main samples


# ── Reward Shaping ─────────────────────────────────────────────────────────
@dataclass
class RewardConfig:
    # Length targets (in tokens) per difficulty bucket
    # Tuned for a 1.5B distill model with 1024 max_new_tokens
    length_targets: dict = field(default_factory=lambda: {
        "easy": 80,
        "medium": 250,
        "hard": 600,
    })
    # Overthinking penalty weight per difficulty (higher = punish more)
    overthink_alpha: dict = field(default_factory=lambda: {
        "easy": 0.002,
        "medium": 0.001,
        "hard": 0.0003,
    })
    # Underthinking penalty weight per difficulty (0 = don't punish short on easy)
    underthink_beta: dict = field(default_factory=lambda: {
        "easy": 0.0,
        "medium": 0.0005,
        "hard": 0.001,
    })
    # Correctness reward weight
    correctness_weight: float = 1.0
    # Judge quality score weight (0-1 from LLM judge)
    quality_weight: float = 0.5


# ── Training ───────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: list = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])

    # DPO
    dpo_beta: float = 0.1
    dpo_lr: float = 1e-5            # balanced: high enough to learn, not overfit
    dpo_epochs: int = 2             # 2 epochs to avoid overfitting
    dpo_batch_size: int = 1
    dpo_grad_accum: int = 2         # lower = more gradient steps
    dpo_max_length: int = 1536
    dpo_warmup_ratio: float = 0.1

    # SFT warmup
    sft_lr: float = 2e-5
    sft_epochs: int = 2
    sft_batch_size: int = 2
    sft_grad_accum: int = 4

    # General
    gradient_checkpointing: bool = True
    save_total_limit: int = 3       # keep last N checkpoints
    dataloader_num_workers: int = 0     # 0 for Windows (no multiprocessing issues)
    group_by_length: bool = True


# ── Outer Loop ─────────────────────────────────────────────────────────────
@dataclass
class LoopConfig:
    max_iterations: int = 5
    prompts_per_iteration: int = 30    # all 30 builtin prompts
    # Safety gates
    min_heldout_improvement: float = 0.01  # min win-rate delta to promote
    max_length_growth_ratio: float = 3.0   # stop if mean length grows >3x
    enable_diversity_check: bool = True
    min_pass_at_k: float = 0.15    # stop if pass@K drops below this
    # Judge
    judge_mode: str = "local"      # "local" | "api" | "rule"
    judge_api_provider: str = ""   # "openai" | "anthropic" | "gemini"
    judge_api_key_env: str = ""    # env var name for API key
    # Audit
    audit_every_n_iters: int = 1   # run external audit every N iters
    audit_anchor_set_size: int = 50
    # Eval speed: skip pass@k on non-boundary iterations
    pass_at_k_k: int = 2               # k for pass@k (lower = faster)
    pass_at_k_only_boundary: bool = True  # only run pass@k on iter 0 and last


# ── Defaults ───────────────────────────────────────────────────────────────
rollout_cfg = RolloutConfig()
reward_cfg = RewardConfig()
train_cfg = TrainConfig()
loop_cfg = LoopConfig()
