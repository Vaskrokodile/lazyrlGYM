"""
lazyRLGYM — Difficulty estimation without ground truth
=======================================================
Estimates prompt difficulty from sample consistency across multiple
responses to the same prompt (arXiv:2402.13904), or from the variance of
judge scores across samples.

No ground-truth labels are required — only the model's own behaviour.
"""
from __future__ import annotations

import re
import statistics
from difflib import SequenceMatcher
from typing import Sequence

from config import END_THINK_TOKEN, THINK_TOKEN

__all__ = [
    "estimate_difficulty",
    "estimate_difficulty_from_scores",
    "classify_difficulty",
    "extract_final_answer",
    "consistency_score",
]


# ── Answer extraction ────────────────────────────────────────────────────────

_ANSWER_PATTERNS = [
    # "The answer is 42" / "Final answer: 42"
    re.compile(r"(?:final\s*answer|the\s*answer\s*is|answer\s*[:=])\s*\**\s*([^\n]+)", re.IGNORECASE),
    # \boxed{...} (LaTeX)
    re.compile(r"\\boxed\{([^{}]*)\}"),
    # "#### 42" (GSM8K-style)
    re.compile(r"####\s*([^\n]+)"),
]


def extract_final_answer(response: str) -> str:
    """Extract the final answer from a model response.

    Strategy (in priority order):
      1. Text after the closing </think> tag (the "answer" portion).
      2. The last \\boxed{...} occurrence.
      3. The last "answer is / final answer:" occurrence.
      4. The last non-empty line of the response.

    Returns a normalized, stripped string.
    """
    text = response.strip()
    if not text:
        return ""

    # 1. Prefer content after the thinking block.
    if END_THINK_TOKEN in text:
        after = text.rsplit(END_THINK_TOKEN, 1)[1].strip()
        if after:
            # If the post-think text itself contains a boxed/pattern, use it.
            extracted = _match_pattern(after)
            if extracted:
                return _normalize(extracted)
            # Otherwise take the last non-empty line of the post-think text.
            last_line = _last_non_empty_line(after)
            if last_line:
                return _normalize(last_line)

    # 2/3. Pattern-based extraction over the full response.
    extracted = _match_pattern(text)
    if extracted:
        return _normalize(extracted)

    # 4. Fallback: last non-empty line.
    last_line = _last_non_empty_line(text)
    return _normalize(last_line) if last_line else ""


def _match_pattern(text: str) -> str | None:
    """Return the last match from any of the answer patterns, or None."""
    last_match: str | None = None
    last_pos = -1
    for pat in _ANSWER_PATTERNS:
        for m in pat.finditer(text):
            if m.start() > last_pos:
                last_pos = m.start()
                last_match = m.group(1)
    return last_match


def _last_non_empty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line:
            return line
    return ""


def _normalize(s: str) -> str:
    """Normalize an extracted answer for comparison."""
    s = s.strip().strip("*").strip("`").strip()
    # Collapse internal whitespace.
    s = re.sub(r"\s+", " ", s)
    # Strip surrounding LaTeX $ ... $.
    s = s.strip("$")
    # Strip trailing period.
    if s.endswith("."):
        s = s[:-1].strip()
    return s.lower()


# ── Similarity / consistency ─────────────────────────────────────────────────

def _text_similarity(a: str, b: str) -> float:
    """Normalized text similarity in [0, 1].

    Uses an exact-match fast path, then a numeric fast path (so that
    "42" vs "42.0" counts as a match), and finally a SequenceMatcher
    ratio over the normalized strings.
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0

    # Numeric comparison: extract the first number from each and compare.
    na = _first_number(a)
    nb = _first_number(b)
    if na is not None and nb is not None:
        return 1.0 if abs(na - nb) < 1e-6 else 0.0

    return SequenceMatcher(None, a, b).ratio()


def _first_number(s: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return float(m.group(0)) if m else None


def consistency_score(answers: Sequence[str]) -> float:
    """Mean pairwise similarity over a list of extracted answers.

    Returns 1.0 when all answers are identical (or there is only one),
    and ~0.0 when answers are completely disjoint.
    """
    answers = [a for a in answers if a]
    n = len(answers)
    if n == 0:
        return 0.0
    if n == 1:
        return 1.0

    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += _text_similarity(answers[i], answers[j])
            pairs += 1
    return total / pairs if pairs else 0.0


# ── Bucketing ────────────────────────────────────────────────────────────────

def classify_difficulty(consistency: float) -> str:
    """Map a consistency score in [0, 1] to a difficulty bucket.

    High consistency (model agrees with itself) → easy.
    Low consistency (model disagrees) → hard.
    """
    if consistency > 0.7:
        return "easy"
    if consistency >= 0.35:
        return "medium"
    return "hard"


# ── Public API ───────────────────────────────────────────────────────────────

def estimate_difficulty(
    responses: list[str],
    n_samples: int = 4,
) -> tuple[str, float]:
    """Estimate prompt difficulty from sample consistency.

    Given multiple responses to the same prompt, extracts the final answer
    from each, measures pairwise agreement, and buckets the prompt as
    "easy" / "medium" / "hard".

    Args:
        responses: Model responses (>=1) to the same prompt.
        n_samples: Hint for how many samples were intended; if more
            responses than ``n_samples`` are provided, only the first
            ``n_samples`` are used. If fewer are provided, all are used.

    Returns:
        (difficulty_bucket, confidence) where confidence is the
        consistency score in [0, 1].
    """
    if not responses:
        return "medium", 0.0

    # Use at most n_samples responses (the rest are for judging/training).
    subset = list(responses)[:max(1, n_samples)]
    answers = [extract_final_answer(r) for r in subset]

    # If every answer is empty (model produced nothing parseable), we
    # cannot measure consistency — treat as hard (model is confused).
    if not any(answers):
        return "hard", 0.0

    consistency = consistency_score(answers)
    bucket = classify_difficulty(consistency)
    return bucket, round(consistency, 4)


def estimate_difficulty_from_scores(
    scores: list[float],
) -> tuple[str, float]:
    """Estimate difficulty from the variance of judge scores.

    Low variance in scores across samples → the prompt is easy (model
    reliably succeeds or reliably fails in a consistent way).
    High variance → the prompt is hard (model is uncertain / inconsistent).

    Args:
        scores: Per-sample judge scores in [0, 1].

    Returns:
        (difficulty_bucket, confidence) where confidence reflects how
        decisive the variance signal is (1 - normalized_stdev).
    """
    scores = [float(s) for s in scores if s is not None]
    n = len(scores)
    if n == 0:
        return "medium", 0.0
    if n == 1:
        # A single score gives no variance signal.
        return "medium", 0.5

    mean = statistics.fmean(scores)
    stdev = statistics.pstdev(scores) if n > 1 else 0.0

    # Normalize stdev to [0, 1] range (max stdev of [0,1] values is 0.5).
    norm_stdev = min(stdev / 0.5, 1.0)

    # Convert variance → consistency-like score.
    # High variance  → low consistency  → hard.
    # Low variance   → high consistency → easy.
    consistency = 1.0 - norm_stdev

    # If the model reliably *fails* (all scores low) that is also "hard",
    # but the variance signal already captures the uncertainty. We keep
    # the pure-variance mapping consistent with classify_difficulty.
    bucket = classify_difficulty(consistency)
    return bucket, round(consistency, 4)
