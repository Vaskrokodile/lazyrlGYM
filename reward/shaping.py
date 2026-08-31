"""
lazyRLGYM — Length × difficulty reward shaping
==============================================
Implements the reward formula from the research brief (§3):

    L_target          = length_targets[difficulty]
    overthink_penalty = max(0, L - L_target) * overthink_alpha[difficulty]
    underthink_penalty= max(0, L_target - L) * underthink_beta[difficulty]
    reward            = correctness_weight * correctness
                      + quality_weight * quality
                      - overthink_penalty
                      - underthink_penalty

This is the API-only analog of LASER-D / LEASH / Kimi k1.5 length rewards:
it punishes "too much" thinking on easy prompts and "too little" thinking
on hard prompts.
"""
from __future__ import annotations

import statistics
from typing import Sequence

from config import END_THINK_TOKEN, RewardConfig, THINK_TOKEN

__all__ = [
    "compute_shaped_reward",
    "compute_all_rewards",
    "get_length_target",
    "analyze_length_distribution",
    "count_thinking_tokens",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def count_thinking_tokens(rollout: dict) -> int:
    """Count the number of tokens in the thinking portion of a rollout.

    A rollout dict is expected to contain either:
      - ``thinking`` (the extracted thinking text), or
      - ``response`` / ``text`` (the full response, from which the
        ``<think>...</think>`` block is extracted).

    Token counting uses a whitespace heuristic so it works without a
    tokenizer; callers that have a tokenizer available may override the
    ``thinking_tokens`` field directly on the rollout.
    """
    if "thinking_tokens" in rollout and rollout["thinking_tokens"] is not None:
        try:
            return int(rollout["thinking_tokens"])
        except (TypeError, ValueError):
            pass

    thinking = rollout.get("thinking")
    if thinking is None:
        response = rollout.get("response") or rollout.get("text") or ""
        thinking = _extract_thinking(response)

    if not thinking:
        return 0
    # Whitespace-based token estimate (~0.75 words/token for English/code).
    return len(thinking.split())


def _extract_thinking(response: str) -> str:
    """Extract the content of the first <think>...</think> block."""
    if THINK_TOKEN not in response:
        return ""
    if END_THINK_TOKEN in response:
        start = response.find(THINK_TOKEN) + len(THINK_TOKEN)
        end = response.find(END_THINK_TOKEN, start)
        if end > start:
            return response[start:end]
    # Unterminated thinking block — take everything after the open tag.
    start = response.find(THINK_TOKEN) + len(THINK_TOKEN)
    return response[start:]


def get_length_target(difficulty: str, cfg: RewardConfig) -> int:
    """Return the target thinking-token length for a difficulty bucket."""
    if not isinstance(difficulty, str) or difficulty not in cfg.length_targets:
        # Unknown bucket → fall back to the medium target.
        difficulty = "medium"
    return int(cfg.length_targets[difficulty])


def _penalty_weights(difficulty: str, cfg: RewardConfig) -> tuple[float, float]:
    """Return (overthink_alpha, underthink_beta) for a difficulty bucket."""
    if not isinstance(difficulty, str) or difficulty not in cfg.overthink_alpha:
        difficulty = "medium"
    alpha = float(cfg.overthink_alpha.get(difficulty, cfg.overthink_alpha["medium"]))
    beta = float(cfg.underthink_beta.get(difficulty, cfg.underthink_beta["medium"]))
    return alpha, beta


# ── Core reward ──────────────────────────────────────────────────────────────

def compute_shaped_reward(
    correctness: float,
    quality: float,
    thinking_tokens: int,
    difficulty: str,
    cfg: RewardConfig,
) -> float:
    """Compute the shaped reward for a single rollout.

    Args:
        correctness: Correctness score in [0, 1].
        quality: Judge quality score in [0, 1].
        thinking_tokens: Number of tokens in the thinking trace.
        difficulty: One of "easy" / "medium" / "hard".
        cfg: A RewardConfig instance.

    Returns:
        The scalar shaped reward.
    """
    correctness = float(correctness)
    quality = float(quality)
    L = int(thinking_tokens)
    difficulty = difficulty if difficulty in cfg.length_targets else "medium"

    L_target = get_length_target(difficulty, cfg)
    alpha, beta = _penalty_weights(difficulty, cfg)

    overthink = max(0.0, L - L_target) * alpha
    underthink = max(0.0, L_target - L) * beta

    reward = (
        float(cfg.correctness_weight) * correctness
        + float(cfg.quality_weight) * quality
        - overthink
        - underthink
    )
    return reward


def compute_all_rewards(
    rollouts: list[dict],
    difficulties: list[str],
    judge_scores: list[float],
    correctness_scores: list[float],
    cfg: RewardConfig,
) -> list[float]:
    """Batch reward computation.

    Args:
        rollouts: List of rollout dicts (used for thinking-token counts).
        difficulties: Per-prompt difficulty bucket (one per rollout).
        judge_scores: Per-rollout quality scores in [0, 1].
        correctness_scores: Per-rollout correctness scores in [0, 1].
        cfg: A RewardConfig instance.

    Returns:
        List of shaped rewards, one per rollout.
    """
    n = len(rollouts)
    if not (len(difficulties) == n and len(judge_scores) == n and len(correctness_scores) == n):
        raise ValueError(
            f"Length mismatch: rollouts={n}, difficulties={len(difficulties)}, "
            f"judge_scores={len(judge_scores)}, correctness_scores={len(correctness_scores)}"
        )

    rewards: list[float] = []
    for rollout, difficulty, quality, correctness in zip(
        rollouts, difficulties, judge_scores, correctness_scores, strict=True
    ):
        L = count_thinking_tokens(rollout)
        rewards.append(compute_shaped_reward(correctness, quality, L, difficulty, cfg))
    return rewards


# ── Diagnostics ──────────────────────────────────────────────────────────────

def analyze_length_distribution(
    rollouts: list[dict],
    difficulties: list[str],
) -> dict:
    """Return length statistics per difficulty bucket.

    Produces:
      - per-bucket: count, mean, median, min, max, std
      - a coarse histogram (bucketed by powers of 2) per difficulty
      - an overall summary
    """
    if len(difficulties) != len(rollouts):
        raise ValueError(
            f"Length mismatch: rollouts={len(rollouts)}, difficulties={len(difficulties)}"
        )

    buckets: dict[str, list[int]] = {"easy": [], "medium": [], "hard": []}
    for rollout, difficulty in zip(rollouts, difficulties, strict=True):
        bucket = difficulty if difficulty in buckets else "medium"
        buckets[bucket].append(count_thinking_tokens(rollout))

    result: dict = {}
    all_lengths: list[int] = []
    for bucket, lengths in buckets.items():
        if not lengths:
            result[bucket] = {
                "count": 0,
                "mean": 0.0,
                "median": 0.0,
                "min": 0,
                "max": 0,
                "std": 0.0,
                "histogram": {},
            }
            continue
        all_lengths.extend(lengths)
        result[bucket] = {
            "count": len(lengths),
            "mean": round(statistics.fmean(lengths), 2),
            "median": statistics.median(lengths),
            "min": min(lengths),
            "max": max(lengths),
            "std": round(statistics.pstdev(lengths), 2) if len(lengths) > 1 else 0.0,
            "histogram": _histogram(lengths),
        }

    result["overall"] = {
        "count": len(all_lengths),
        "mean": round(statistics.fmean(all_lengths), 2) if all_lengths else 0.0,
        "median": statistics.median(all_lengths) if all_lengths else 0.0,
        "min": min(all_lengths) if all_lengths else 0,
        "max": max(all_lengths) if all_lengths else 0,
        "std": round(statistics.pstdev(all_lengths), 2) if len(all_lengths) > 1 else 0.0,
        "histogram": _histogram(all_lengths) if all_lengths else {},
    }
    return result


def _histogram(lengths: Sequence[int]) -> dict[str, int]:
    """Coarse log-2 bucketed histogram of token lengths."""
    bins = [
        (0, 64),
        (64, 128),
        (128, 256),
        (256, 512),
        (512, 1024),
        (1024, 2048),
        (2048, 4096),
        (4096, 8192),
        (8192, float("inf")),
    ]
    labels = [
        "0-64",
        "64-128",
        "128-256",
        "256-512",
        "512-1024",
        "1024-2048",
        "2048-4096",
        "4096-8192",
        "8192+",
    ]
    hist = {label: 0 for label in labels}
    for L in lengths:
        for (lo, hi), label in zip(bins, labels, strict=True):
            if lo <= L < hi:
                hist[label] += 1
                break
    return hist
