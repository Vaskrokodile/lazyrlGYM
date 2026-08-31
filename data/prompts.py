"""
lazyRLGYM — Prompt Bank
=======================
Curated prompts of varying difficulty for the efficient-reasoning RL loop.

Difficulty buckets drive the reward shaping in ``config.RewardConfig``:
  * easy   -> short answers expected, heavy overthinking penalty
  * medium -> moderate reasoning, balanced penalties
  * hard   -> long reasoning allowed, underthinking penalised

This module also exposes ``load_fable5_prompts`` which pulls complex agentic
user tasks from the Glint-Research/Fable-5-traces dataset on HuggingFace and
treats them as ``hard`` prompts (they are open-ended, multi-step coding tasks
taken from real Claude Code sessions).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# Ensure the project root (parent of this package) is importable as ``config``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DATA_CACHE_DIR  # noqa: E402

# ── Token markers (kept in sync with config) ────────────────────────────────
THINK_TOKEN = "<think>"
END_THINK_TOKEN = "</think>"

# ── HuggingFace dataset reference ───────────────────────────────────────────
FABLE5_REPO = "Glint-Research/Fable-5-traces"
FABLE5_MERGED_FILE = "fable5_cot_merged.jsonl"
FABLE5_CACHE_PATH = DATA_CACHE_DIR / "fable5_prompts.json"

# Maximum number of Fable-5 prompts to extract (keeps the pilot lightweight).
FABLE5_MAX_PROMPTS = 500


# ── Default prompt bank ─────────────────────────────────────────────────────
# Each entry: {"id", "text", "difficulty_hint", "category"}
_DEFAULT_PROMPTS: list[dict] = [
    # ── Easy (10) ───────────────────────────────────────────────────────────
    {"id": "easy-001", "text": "What is 2 + 3?", "difficulty_hint": "easy", "category": "arithmetic", "expected_answer": "5"},
    {"id": "easy-002", "text": "What is 7 multiplied by 6?", "difficulty_hint": "easy", "category": "arithmetic", "expected_answer": "42"},
    {"id": "easy-003", "text": "What is the capital of France?", "difficulty_hint": "easy", "category": "world_facts", "expected_answer": "Paris"},
    {"id": "easy-004", "text": "How many days are there in a leap year?", "difficulty_hint": "easy", "category": "world_facts", "expected_answer": "366"},
    {"id": "easy-005", "text": "Write a Python function to reverse a string.", "difficulty_hint": "easy", "category": "coding"},
    {"id": "easy-006", "text": "Write a function that returns the maximum of two numbers.", "difficulty_hint": "easy", "category": "coding"},
    {"id": "easy-007", "text": "What is the chemical symbol for water?", "difficulty_hint": "easy", "category": "science", "expected_answer": "H2O"},
    {"id": "easy-008", "text": "Convert 100 degrees Celsius to Fahrenheit.", "difficulty_hint": "easy", "category": "arithmetic", "expected_answer": "212"},
    {"id": "easy-009", "text": "List the primary colors.", "difficulty_hint": "easy", "category": "world_facts", "expected_answer": "red, blue, yellow"},
    {"id": "easy-010", "text": "Write a one-line Python snippet to sort a list of integers.", "difficulty_hint": "easy", "category": "coding"},

    # ── Medium (10) ─────────────────────────────────────────────────────────
    {"id": "med-001", "text": "A train travels 60 km in 45 minutes. What is its average speed in km/h, and how long would it take to cover 200 km at that speed?", "difficulty_hint": "medium", "category": "multi_step_math", "expected_answer": "80 km/h, 150 minutes"},
    {"id": "med-002", "text": "Explain how a hash table works, including how collisions are handled with separate chaining.", "difficulty_hint": "medium", "category": "explanation"},
    {"id": "med-003", "text": "Write a Python function that checks whether a string is a valid palindrome, ignoring spaces, punctuation, and case.", "difficulty_hint": "medium", "category": "coding"},
    {"id": "med-004", "text": "Given a list of integers, return the two indices whose values sum to a target. Assume exactly one solution exists.", "difficulty_hint": "medium", "category": "coding"},
    {"id": "med-005", "text": "Explain the difference between supervised learning and reinforcement learning with one concrete example of each.", "difficulty_hint": "medium", "category": "explanation"},
    {"id": "med-006", "text": "Write a Python function to compute the n-th Fibonacci number using memoization.", "difficulty_hint": "medium", "category": "coding"},
    {"id": "med-007", "text": "A shop gives a 15% discount on items over $50 and a 5% discount otherwise. Compute the final price for a $72 item including 8% tax.", "difficulty_hint": "medium", "category": "multi_step_math", "expected_answer": "66.08"},
    {"id": "med-008", "text": "Summarize the plot of Romeo and Juliet in five sentences.", "difficulty_hint": "medium", "category": "explanation"},
    {"id": "med-009", "text": "Write a SQL query that finds the top 3 customers by total order value from an `orders` table with columns `customer_id` and `amount`.", "difficulty_hint": "medium", "category": "coding"},
    {"id": "med-010", "text": "Explain what Big-O notation is and give the time complexity of binary search with a brief justification.", "difficulty_hint": "medium", "category": "explanation"},

    # ── Hard (10) ───────────────────────────────────────────────────────────
    {"id": "hard-001", "text": "Design an LRU cache supporting get and put operations in O(1) time. Provide a full implementation in Python with a short complexity analysis.", "difficulty_hint": "hard", "category": "algorithm_design"},
    {"id": "hard-002", "text": "Design the system architecture for a URL shortener service that handles 100 million URLs and 200 reads per second. Cover data model, scaling, caching, and failure modes.", "difficulty_hint": "hard", "category": "system_design"},
    {"id": "hard-003", "text": "Implement Dijkstra's shortest-path algorithm from scratch in Python using a binary heap. Include input validation and a worked example.", "difficulty_hint": "hard", "category": "algorithm_design"},
    {"id": "hard-004", "text": "Design a rate limiter that supports both token-bucket and sliding-window strategies, is distributed across multiple nodes, and explains the consistency tradeoffs.", "difficulty_hint": "hard", "category": "system_design"},
    {"id": "hard-005", "text": "Write a Python program that solves the N-Queens puzzle for N=8 using backtracking and prints every distinct solution as a board.", "difficulty_hint": "hard", "category": "algorithm_design"},
    {"id": "hard-006", "text": "Design a key-value store with eventual consistency, support for conflict resolution via vector clocks, and a partition-tolerant replication strategy. Justify each design choice.", "difficulty_hint": "hard", "category": "system_design"},
    {"id": "hard-007", "text": "Implement a thread-safe bounded queue in Python with timeout-aware put/get operations and a clean shutdown mechanism. Include a short correctness argument.", "difficulty_hint": "hard", "category": "multi_constraint_coding"},
    {"id": "hard-008", "text": "Given a weighted directed graph, implement both Bellman-Ford and Floyd-Warshall, compare their asymptotic complexity, and describe when each is preferable.", "difficulty_hint": "hard", "category": "algorithm_design"},
    {"id": "hard-009", "text": "Design a notification system that delivers emails, SMS, and push notifications with retry, deduplication, priority queues, and user preference rules. Outline the data flow and failure handling.", "difficulty_hint": "hard", "category": "system_design"},
    {"id": "hard-010", "text": "Implement a mini regex engine in Python supporting literals, '.', '*', '+', '?', character classes, and anchors. Include tests for each feature.", "difficulty_hint": "hard", "category": "multi_constraint_coding"},
]


def get_prompts(n: int, difficulty: Optional[str] = None) -> list[dict]:
    """Return up to ``n`` prompts from the default bank.

    Parameters
    ----------
    n:
        Number of prompts to return.
    difficulty:
        Optional filter; one of ``"easy"``, ``"medium"``, ``"hard"``.
        If ``None``, prompts are sampled across all difficulties (cycling
        through easy -> medium -> hard for balanced coverage).

    Returns
    -------
    list[dict]
        Each dict has keys ``id``, ``text``, ``difficulty_hint``, ``category``.
    """
    if n <= 0:
        return []

    valid_diffs = {"easy", "medium", "hard"}
    if difficulty is not None and difficulty not in valid_diffs:
        raise ValueError(
            f"difficulty must be one of {valid_diffs} or None, got {difficulty!r}"
        )

    if difficulty is not None:
        pool = [p for p in _DEFAULT_PROMPTS if p["difficulty_hint"] == difficulty]
    else:
        # Interleave difficulties for balanced coverage.
        buckets: dict[str, list[dict]] = {"easy": [], "medium": [], "hard": []}
        for p in _DEFAULT_PROMPTS:
            buckets[p["difficulty_hint"]].append(p)
        pool = []
        idx = 0
        while len(pool) < len(_DEFAULT_PROMPTS):
            added = False
            for diff in ("easy", "medium", "hard"):
                if idx < len(buckets[diff]):
                    pool.append(buckets[diff][idx])
                    added = True
            if not added:
                break
            idx += 1

    return pool[:n]


# ── Fable-5 prompt extraction ───────────────────────────────────────────────
# Markers used in the flattened ``context`` transcript field.
_USER_LINE_RE = re.compile(r"^USER:\s?(.*)$", re.DOTALL)
# Noise prefixes that indicate a USER line is not a real task prompt.
_NOISE_PREFIXES = (
    "<local-command-caveat>",
    "<local-command-stdout>",
    "<command-name>",
    "<command-message>",
    "<command-args>",
    "<command-stdout>",
    "<command-error>",
)
_NOISE_TOKENS = ("/model", "/clear", "/help", "/cost", "/compact", "/status")


def _is_noise_user_line(text: str) -> bool:
    """Return True if a USER: line is CLI noise rather than a real task."""
    stripped = text.strip()
    if not stripped:
        return True
    low = stripped.lower()
    if any(stripped.startswith(p) for p in _NOISE_PREFIXES):
        return True
    if any(tok in low for tok in _NOISE_TOKENS):
        return True
    # Very short lines are usually echoes/acknowledgements, not tasks.
    if len(stripped) < 12:
        return True
    return False


def _extract_user_prompt(context: str) -> Optional[str]:
    """Pull the most meaningful user task prompt out of a Fable-5 transcript.

    The ``context`` field is a flattened transcript with ``USER:`` and
    ``ASSISTANT`` markers. We collect all real USER lines (filtering out
    CLI command noise) and return the longest one, since the primary task
    description is typically the most detailed user message.
    """
    if not context:
        return None

    candidates: list[str] = []
    for raw_line in context.split("\n"):
        m = _USER_LINE_RE.match(raw_line)
        if not m:
            continue
        text = m.group(1).strip()
        if _is_noise_user_line(text):
            continue
        candidates.append(text)

    if not candidates:
        return None
    # Longest meaningful line is usually the actual task spec.
    return max(candidates, key=len)


def _download_fable5_merged() -> Optional[Path]:
    """Download (or reuse the cached) ``fable5_cot_merged.jsonl`` file.

    Returns the local path to the JSONL file, or ``None`` on failure.
    Network errors are handled gracefully so callers can fall back to the
    default prompt bank.
    """
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - import guard
        print(f"[prompts] huggingface_hub unavailable: {exc}")
        return None

    try:
        local_path = hf_hub_download(
            repo_id=FABLE5_REPO,
            filename=FABLE5_MERGED_FILE,
            repo_type="dataset",
        )
        return Path(local_path)
    except Exception as exc:
        print(f"[prompts] Failed to download Fable-5-traces: {exc}")
        return None


def load_fable5_prompts(
    force_refresh: bool = False, max_prompts: int = FABLE5_MAX_PROMPTS
) -> list[dict]:
    """Load complex agentic prompts from the Fable-5-traces dataset.

    Each returned dict has the standard shape::

        {"id", "text", "difficulty_hint": "hard", "category"}

    Results are cached to ``DATA_CACHE_DIR / "fable5_prompts.json"``. If the
    dataset cannot be downloaded (e.g. no network), an empty list is returned
    so the caller can fall back to ``get_prompts``.

    Parameters
    ----------
    force_refresh:
        Re-download and re-extract even if a cache file exists.
    max_prompts:
        Maximum number of prompts to extract (default 500 for the pilot).
    """
    if not force_refresh and FABLE5_CACHE_PATH.exists():
        try:
            with open(FABLE5_CACHE_PATH, "r", encoding="utf-8") as fh:
                cached = json.load(fh)
            if isinstance(cached, list) and cached:
                return cached[:max_prompts]
        except Exception as exc:
            print(f"[prompts] Cache read failed ({exc}); re-extracting.")

    merged_path = _download_fable5_merged()
    if merged_path is None or not merged_path.exists():
        print("[prompts] Fable-5 download unavailable; returning empty list.")
        return []

    prompts: list[dict] = []
    seen: set[str] = set()
    try:
        with open(merged_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                context = record.get("context", "") or ""
                prompt_text = _extract_user_prompt(context)
                if not prompt_text:
                    continue
                # Deduplicate by prompt text.
                key = prompt_text.strip().lower()
                if key in seen:
                    continue
                seen.add(key)
                uid = record.get("uid") or f"fable5-{len(prompts):04d}"
                prompts.append(
                    {
                        "id": f"fable5-{uid}",
                        "text": prompt_text,
                        "difficulty_hint": "hard",
                        "category": "agentic_coding",
                    }
                )
                if len(prompts) >= max_prompts:
                    break
    except Exception as exc:
        print(f"[prompts] Error reading Fable-5 file: {exc}")
        # Return whatever we managed to extract so far.

    # Persist cache.
    try:
        FABLE5_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FABLE5_CACHE_PATH, "w", encoding="utf-8") as fh:
            json.dump(prompts, fh, ensure_ascii=False, indent=2)
    except Exception as exc:
        print(f"[prompts] Cache write failed: {exc}")

    return prompts


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import argparse

    parser = argparse.ArgumentParser(description="Inspect the lazyRLGYM prompt bank.")
    parser.add_argument("--n", type=int, default=5, help="number of prompts to print")
    parser.add_argument(
        "--difficulty",
        default=None,
        choices=["easy", "medium", "hard"],
        help="filter by difficulty",
    )
    parser.add_argument(
        "--fable5", action="store_true", help="load Fable-5 prompts instead of defaults"
    )
    args = parser.parse_args()

    if args.fable5:
        ps = load_fable5_prompts()
    else:
        ps = get_prompts(args.n, difficulty=args.difficulty)
    print(f"Loaded {len(ps)} prompts:")
    for p in ps:
        print(f"  [{p['id']}] ({p['difficulty_hint']}/{p['category']}) {p['text'][:80]}")
