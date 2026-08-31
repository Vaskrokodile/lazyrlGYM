"""
lazyRLGYM — Preference Pair Construction for DPO
================================================
Turns RL rollouts + reward scores into preference pairs for Direct Preference
Optimization (DPO).

The efficient-reasoning objective rewards:
  * concise answers on easy prompts   (punish overthinking)
  * thorough reasoning on hard prompts (punish underthinking)

For each prompt we group its N rollouts, pick the highest-scoring response as
``chosen`` and the lowest-scoring as ``rejected``. Pairs where chosen and
rejected are tied (no learnable signal) are dropped. The resulting pairs are
formatted for TRL's ``DPOTrainer`` and persisted as JSONL.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

# Ensure the project root (parent of this package) is importable as ``config``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DATA_CACHE_DIR  # noqa: E402

# Default output location for the preference dataset.
DEFAULT_PREF_PATH = DATA_CACHE_DIR / "preference_pairs.jsonl"


def build_preference_pairs(
    rollouts: list[dict],
    reward_scores: list[float],
    min_score_delta: float = 0.05,
    max_pairs_per_prompt: int = 3,
) -> list[dict]:
    """Construct DPO preference pairs from rollouts and their reward scores.

    Generates multiple pairs per prompt by pairing high-score rollouts with
    low-score rollouts (best vs worst, 2nd best vs 2nd worst, etc.).

    Parameters
    ----------
    rollouts:
        List of rollout dicts.
    reward_scores:
        Parallel list of float reward scores (one per rollout).
    min_score_delta:
        Minimum score difference between chosen and rejected. Pairs with
        smaller deltas are dropped (no clear preference signal).
    max_pairs_per_prompt:
        Maximum number of pairs to generate per prompt.

    Returns
    -------
    list[dict]
        Multiple preference pairs per prompt. Each dict::

            {"prompt": str, "chosen": str, "rejected": str,
             "chosen_score": float, "rejected_score": float,
             "difficulty": str}
    """
    if len(rollouts) != len(reward_scores):
        raise ValueError(
            f"rollouts and reward_scores must be the same length, "
            f"got {len(rollouts)} and {len(reward_scores)}"
        )

    # Group (score, rollout) tuples by prompt_id, preserving insertion order.
    groups: dict[str, list[tuple[float, dict]]] = defaultdict(list)
    for rollout, score in zip(rollouts, reward_scores):
        pid = rollout.get("prompt_id")
        if pid is None:
            continue
        groups[pid].append((float(score), rollout))

    pairs: list[dict] = []
    for pid, items in groups.items():
        if len(items) < 2:
            continue

        # Sort by score descending.
        items.sort(key=lambda x: x[0], reverse=True)

        # Generate pairs: best vs worst, 2nd best vs 2nd worst, etc.
        n_pairs = min(max_pairs_per_prompt, len(items) // 2)
        seen_responses: set[str] = set()

        for pidx in range(n_pairs):
            chosen_score, chosen = items[pidx]
            rejected_score, rejected = items[-(pidx + 1)]

            # Skip if not enough score delta.
            if chosen_score - rejected_score < min_score_delta:
                continue

            # Skip identical responses.
            chosen_resp = chosen.get("response", "")
            rejected_resp = rejected.get("response", "")
            if chosen_resp == rejected_resp:
                continue
            # Skip if we've already seen either response for this prompt.
            key = chosen_resp[:100] + "|" + rejected_resp[:100]
            if key in seen_responses:
                continue
            seen_responses.add(key)

            pairs.append(
                {
                    "prompt": chosen.get("prompt", ""),
                    "chosen": chosen_resp,
                    "rejected": rejected_resp,
                    "chosen_score": chosen_score,
                    "rejected_score": rejected_score,
                    "difficulty": chosen.get("difficulty", "medium"),
                }
            )

    return pairs


def format_for_dpo(pair: dict) -> dict:
    """Format a preference pair for TRL's ``DPOTrainer``.

    Returns::

        {"prompt": str, "chosen": str, "rejected": str}

    Extra fields (scores, difficulty) are dropped because DPOTrainer only
    consumes prompt/chosen/rejected.
    """
    return {
        "prompt": pair["prompt"],
        "chosen": pair["chosen"],
        "rejected": pair["rejected"],
    }


def save_preference_dataset(pairs: list[dict], path: Path) -> None:
    """Save preference pairs as a JSONL file (one JSON object per line)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for pair in pairs:
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")


def load_preference_dataset(path: Path) -> list[dict]:
    """Load a JSONL preference dataset produced by :func:`save_preference_dataset`.

    Returns
    -------
    list[dict]
        Each dict has keys ``prompt``, ``chosen``, ``rejected``,
        ``chosen_score``, ``rejected_score``, ``difficulty`` (depending on what
        was saved). Missing/malformed lines are skipped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Preference dataset not found: {path}")

    pairs: list[dict] = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[preferences] Skipping malformed line: {exc}")
    return pairs


def build_and_save(
    rollouts: list[dict],
    reward_scores: list[float],
    path: Optional[Path] = None,
) -> tuple[list[dict], Path]:
    """Convenience wrapper: build pairs, save them, and return (pairs, path)."""
    pairs = build_preference_pairs(rollouts, reward_scores)
    out_path = Path(path) if path is not None else DEFAULT_PREF_PATH
    save_preference_dataset(pairs, out_path)
    return pairs, out_path


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    # Build a tiny synthetic example to demonstrate the pipeline.
    demo_rollouts = [
        {"prompt_id": "p1", "prompt": "What is 2+2?", "response": "4", "thinking_tokens": 0, "answer_tokens": 1, "difficulty": "easy"},
        {"prompt_id": "p1", "prompt": "What is 2+2?", "response": "Let me think... 2 plus 2 equals 4.", "thinking_tokens": 12, "answer_tokens": 6, "difficulty": "easy"},
        {"prompt_id": "p2", "prompt": "Explain recursion.", "response": "short", "thinking_tokens": 0, "answer_tokens": 1, "difficulty": "medium"},
        {"prompt_id": "p2", "prompt": "Explain recursion.", "response": "Recursion is when a function calls itself...", "thinking_tokens": 40, "answer_tokens": 30, "difficulty": "medium"},
    ]
    # Easy prompt: concise answer wins (punish overthinking).
    # Medium prompt: thorough answer wins (punish underthinking).
    demo_scores = [1.0, 0.2, 0.1, 0.9]

    pairs = build_preference_pairs(demo_rollouts, demo_scores)
    print(f"Built {len(pairs)} preference pairs:")
    for p in pairs:
        print(f"  [{p['difficulty']}] chosen={p['chosen_score']:.2f} "
              f"rejected={p['rejected_score']:.2f} | {p['prompt'][:40]}")

    out = DATA_CACHE_DIR / "demo_preference_pairs.jsonl"
    save_preference_dataset(pairs, out)
    loaded = load_preference_dataset(out)
    print(f"Saved & reloaded {len(loaded)} pairs from {out}")
