"""
lazyRLGYM — Fable-5 Dataset Loading & SFT Processing
====================================================
Downloads the Glint-Research/Fable-5-traces dataset and converts the long
agentic reasoning traces into (prompt, response) pairs suitable for an SFT
warmup phase before the RL loop.

Dataset note
------------
The Fable-5-traces repo ships a pre-merged ``fable5_cot_merged.jsonl`` file
(one JSON object per assistant turn) with the following schema::

    {
      "uid": "<unique id>",
      "source_file": "<original .jsonl path>",
      "session": "<session id>",
      "model": "claude-fable-5",
      "context": "<flattened transcript up to this turn>",
      "cot": "<chain-of-thought text, no think markers>",
      "output_type": "text" | "tool_use",
      "output": {"text": "..."} | {"tool": "...", "input": {...}},
      "completion": "<full completion string>",
      "origin": "local"
    }

The standard ``datasets.load_dataset`` call currently fails on this repo
because its features use a ``Json`` type not recognised by the installed
``datasets`` version. We therefore try ``datasets`` first and transparently
fall back to a direct ``hf_hub_download`` of the merged JSONL file.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Optional

# Ensure the project root (parent of this package) is importable as ``config``.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import DATA_CACHE_DIR, THINK_TOKEN, END_THINK_TOKEN  # noqa: E402

# ── Constants ───────────────────────────────────────────────────────────────
FABLE5_REPO = "Glint-Research/Fable-5-traces"
FABLE5_MERGED_FILE = "fable5_cot_merged.jsonl"
FABLE5_CACHE_PATH = DATA_CACHE_DIR / "fable5_traces.jsonl"

# Pilot limit: keep the first 500 traces lightweight for the warmup phase.
DEFAULT_TRACE_LIMIT = 500

# Re-use the user-prompt extractor from the prompt bank.
from data.prompts import _extract_user_prompt  # noqa: E402


# ── Download / load ─────────────────────────────────────────────────────────
def _try_load_dataset(limit: int) -> Optional[list[dict]]:
    """Attempt ``datasets.load_dataset``; return records or ``None`` on failure."""
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - import guard
        print(f"[dataset] datasets library unavailable: {exc}")
        return None

    try:
        # The merged file is the cleanest single-file view of the dataset.
        ds = load_dataset(FABLE5_REPO, data_files=FABLE5_MERGED_FILE, split="train")
        records = list(ds.select(range(min(limit, len(ds)))))
        return records
    except Exception as exc:
        print(f"[dataset] load_dataset failed ({type(exc).__name__}: {exc}); "
              f"falling back to direct download.")
        return None


def _download_merged_jsonl() -> Optional[Path]:
    """Download the merged JSONL via ``hf_hub_download``; return path or None."""
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - import guard
        print(f"[dataset] huggingface_hub unavailable: {exc}")
        return None

    try:
        local_path = hf_hub_download(
            repo_id=FABLE5_REPO,
            filename=FABLE5_MERGED_FILE,
            repo_type="dataset",
        )
        return Path(local_path)
    except Exception as exc:
        print(f"[dataset] Failed to download Fable-5-traces: {exc}")
        return None


def load_fable5_traces(
    limit: int = DEFAULT_TRACE_LIMIT, force_refresh: bool = False
) -> list[dict]:
    """Download and return Fable-5-traces records as a list of dicts.

    The result is cached locally as JSONL at ``DATA_CACHE_DIR / fable5_traces.jsonl``
    so repeated runs avoid re-downloading. Only the first ``limit`` records are
    kept (default 500) to keep the pilot lightweight.

    Parameters
    ----------
    limit:
        Maximum number of trace records to return.
    force_refresh:
        Ignore the local cache and re-download.

    Returns
    -------
    list[dict]
        Raw trace records (one per assistant turn). Empty list on failure.
    """
    # 1) Try the local cache first.
    if not force_refresh and FABLE5_CACHE_PATH.exists():
        try:
            cached: list[dict] = []
            with open(FABLE5_CACHE_PATH, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        cached.append(json.loads(line))
                    if len(cached) >= limit:
                        break
            if cached:
                return cached
        except Exception as exc:
            print(f"[dataset] Cache read failed ({exc}); re-downloading.")

    # 2) Try the datasets library (preferred path per project requirements).
    records = _try_load_dataset(limit)
    if records is None:
        # 3) Fall back to a direct file download + manual JSONL parse.
        merged_path = _download_merged_jsonl()
        if merged_path is None or not merged_path.exists():
            print("[dataset] All download strategies failed; returning empty list.")
            return []
        records = []
        try:
            with open(merged_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                    if len(records) >= limit:
                        break
        except Exception as exc:
            print(f"[dataset] Error reading merged JSONL: {exc}")

    if not records:
        return []

    # Persist cache (write only up to the requested limit).
    try:
        FABLE5_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(FABLE5_CACHE_PATH, "w", encoding="utf-8") as fh:
            for rec in records[:limit]:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as exc:
        print(f"[dataset] Cache write failed: {exc}")

    return records[:limit]


# ── SFT pair extraction ─────────────────────────────────────────────────────
def _format_output(output: Any, output_type: str) -> str:
    """Render a trace's ``output`` field as plain assistant text.

    * ``text``     -> the message text directly.
    * ``tool_use`` -> a compact, human-readable description of the tool call
                      (kept as text so a chat model can still learn from it).
    """
    if output is None:
        return ""
    if output_type == "text":
        if isinstance(output, dict):
            return str(output.get("text", "")).strip()
        return str(output).strip()
    # tool_use
    if isinstance(output, dict):
        tool = output.get("tool", "unknown")
        inp = output.get("input", {})
        try:
            inp_str = json.dumps(inp, ensure_ascii=False)
        except (TypeError, ValueError):
            inp_str = str(inp)
        return f"[Tool call: {tool}] {inp_str}".strip()
    return str(output).strip()


def _build_assistant_response(cot: str, output: Any, output_type: str) -> str:
    """Build an assistant response string with an R1-style thinking block.

    Layout::

        <think>
        {cot}
        </think>
        {answer}

    If ``cot`` is empty, the thinking block is omitted and only the answer is
    returned (some turns may have no captured reasoning).
    """
    answer = _format_output(output, output_type)
    cot = (cot or "").strip()
    if cot:
        return f"{THINK_TOKEN}\n{cot}\n{END_THINK_TOKEN}\n{answer}".rstrip()
    return answer


def extract_sft_pairs(
    traces: list[dict],
    max_pairs: Optional[int] = None,
    prefer_text: bool = True,
) -> list[dict]:
    """Extract (prompt, response) SFT pairs from Fable-5 trace records.

    Parameters
    ----------
    traces:
        Records as returned by :func:`load_fable5_traces`.
    max_pairs:
        Optional cap on the number of pairs returned.
    prefer_text:
        If True (default), only keep ``text`` turns (clean assistant messages).
        If False, also include ``tool_use`` turns rendered as text.

    Returns
    -------
    list[dict]
        Each dict: ``{"prompt": str, "response": str, "uid": str, "session": str}``.
    """
    pairs: list[dict] = []
    seen: set[str] = set()

    for rec in traces:
        output_type = rec.get("output_type", "")
        if prefer_text and output_type != "text":
            continue
        if output_type not in ("text", "tool_use"):
            continue

        context = rec.get("context", "") or ""
        prompt = _extract_user_prompt(context)
        if not prompt:
            continue

        cot = rec.get("cot", "") or ""
        response = _build_assistant_response(cot, rec.get("output"), output_type)
        if not response:
            continue

        # Skip degenerate pairs.
        if len(prompt) < 12 or len(response) < 8:
            continue

        # Deduplicate by (prompt, response) to avoid near-identical turns.
        key = (prompt.strip().lower(), response.strip().lower())
        if key in seen:
            continue
        seen.add(key)

        pairs.append(
            {
                "prompt": prompt,
                "response": response,
                "uid": rec.get("uid", ""),
                "session": rec.get("session", ""),
            }
        )
        if max_pairs is not None and len(pairs) >= max_pairs:
            break

    return pairs


def format_for_sft(prompt: str, response: str) -> dict:
    """Format a (prompt, response) pair into a chat-style SFT example.

    Returns a dict compatible with HuggingFace's SFTTrainer / chat templates::

        {"messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]}
    """
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def build_sft_dataset(
    limit: int = DEFAULT_TRACE_LIMIT, max_pairs: Optional[int] = None
) -> list[dict]:
    """Convenience helper: load traces and return ready-to-train SFT examples."""
    traces = load_fable5_traces(limit=limit)
    pairs = extract_sft_pairs(traces, max_pairs=max_pairs)
    return [format_for_sft(p["prompt"], p["response"]) for p in pairs]


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import argparse

    parser = argparse.ArgumentParser(description="Inspect Fable-5 SFT extraction.")
    parser.add_argument("--limit", type=int, default=50, help="trace limit")
    parser.add_argument("--max-pairs", type=int, default=5, help="pairs to print")
    parser.add_argument(
        "--include-tools", action="store_true", help="include tool_use turns"
    )
    args = parser.parse_args()

    traces = load_fable5_traces(limit=args.limit)
    print(f"Loaded {len(traces)} traces.")
    pairs = extract_sft_pairs(traces, max_pairs=args.max_pairs, prefer_text=not args.include_tools)
    print(f"Extracted {len(pairs)} SFT pairs (showing up to {args.max_pairs}):")
    for p in pairs:
        print("=" * 70)
        print("PROMPT:", p["prompt"][:160])
        print("RESPONSE:", p["response"][:300])
