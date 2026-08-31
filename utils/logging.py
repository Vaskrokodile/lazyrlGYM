"""
Logging and metrics tracking for the RL training loop.
Tracks per-iteration metrics and writes structured logs.
"""
import json
import time
from pathlib import Path
from typing import Any
from dataclasses import dataclass, field, asdict

from config import LOG_DIR


@dataclass
class IterationMetrics:
    iteration: int = 0
    timestamp: str = ""
    # Rollout
    n_prompts: int = 0
    n_samples: int = 0
    total_tokens_generated: int = 0
    # Difficulty distribution
    n_easy: int = 0
    n_medium: int = 0
    n_hard: int = 0
    # Reward stats
    mean_reward: float = 0.0
    mean_correctness: float = 0.0
    mean_quality: float = 0.0
    # Length stats per difficulty
    mean_length_easy: float = 0.0
    mean_length_medium: float = 0.0
    mean_length_hard: float = 0.0
    # Diversity
    pass_at_1: float = 0.0
    pass_at_k: float = 0.0
    self_bleu: float = 0.0
    # Training
    train_loss: float = 0.0
    # Held-out eval
    heldout_winrate: float = 0.0
    heldout_score: float = 0.0
    # Safety
    judge_truth_gap: float = 0.0
    reward_hacking_flag: bool = False
    # Decision
    promoted: bool = False
    rollback_reason: str = ""


class MetricsLogger:
    def __init__(self):
        self.log_file = LOG_DIR / "metrics.jsonl"
        self.metrics: list[IterationMetrics] = []

    def log_iteration(self, m: IterationMetrics):
        m.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        self.metrics.append(m)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(m)) + "\n")
        self._print_summary(m)

    def _print_summary(self, m: IterationMetrics):
        print(f"\n{'='*60}")
        print(f"  Iteration {m.iteration} — {m.timestamp}")
        print(f"{'='*60}")
        print(f"  Prompts: {m.n_prompts}  Samples: {m.n_samples}")
        print(f"  Difficulty: easy={m.n_easy} med={m.n_medium} hard={m.n_hard}")
        print(f"  Reward: mean={m.mean_reward:.3f}  corr={m.mean_correctness:.3f}  qual={m.mean_quality:.3f}")
        print(f"  Length: easy={m.mean_length_easy:.0f}  med={m.mean_length_medium:.0f}  hard={m.mean_length_hard:.0f}")
        print(f"  Diversity: pass@1={m.pass_at_1:.3f}  pass@k={m.pass_at_k:.3f}")
        print(f"  Train loss: {m.train_loss:.4f}")
        print(f"  Held-out: winrate={m.heldout_winrate:.3f}  score={m.heldout_score:.3f}")
        print(f"  Safety: judge_gap={m.judge_truth_gap:.3f}  hacking={m.reward_hacking_flag}")
        print(f"  Decision: {'PROMOTED' if m.promoted else 'ROLLED BACK'} {m.rollback_reason}")
        print(f"{'='*60}\n")

    def get_history(self) -> list[dict]:
        return [asdict(m) for m in self.metrics]


# Singleton
logger = MetricsLogger()
