"""
lazyRLGYM - Pluggable LLM-as-judge
===================================
Three judge backends sharing a common interface:

  - ``LocalJudge``    : uses a local HF model (can be the policy itself)
                        with a judge prompt; parses structured scores.
  - ``APIJudge``      : uses an external API (OpenAI or Anthropic).
  - ``RuleBasedJudge``: pattern/AST matching for verifiable answers
                        (math, code) - no model required.

Use ``get_judge(mode, **kwargs)`` to obtain a judge instance.
"""
from __future__ import annotations

import ast
import json
import logging
import os
import re
from typing import Any, Optional

from config import END_THINK_TOKEN, MODEL_DTYPE, MODEL_ID, THINK_TOKEN

log = logging.getLogger(__name__)

__all__ = [
    "BaseJudge",
    "LocalJudge",
    "APIJudge",
    "RuleBasedJudge",
    "get_judge",
]


# ── Judge prompt ─────────────────────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """You are a strict but fair judge evaluating a reasoning model's response.

## Task
Evaluate the response on three axes. Respond ONLY with a JSON object.

## Axes
1. **correctness** (0.0-1.0): Is the final answer correct and complete?
   - 1.0 = fully correct, 0.5 = partially correct, 0.0 = wrong.
2. **quality** (0.0-1.0): Is the reasoning sound, well-structured, and free
   of errors, contradictions, or hallucinations? Consider clarity and
   logical validity, NOT length.
3. **length_appropriate** (true/false): Is the thinking length appropriate
   for the difficulty of the prompt? Too long = overthinking; too short =
   underthinking. This is informational only.

## Inputs
### Prompt
{prompt}

### Thinking trace
{thinking}

### Final answer
{answer}

## Output format
Respond with EXACTLY this JSON and nothing else:
{{"correctness": <float 0-1>, "quality": <float 0-1>, "length_appropriate": <true|false>, "reasoning": "<one sentence>"}}
"""


# ── Base class ───────────────────────────────────────────────────────────────

class BaseJudge:
    """Abstract judge interface."""

    def score(self, prompt: str, response: str, thinking: str = "") -> dict:
        """Score a single (prompt, response) pair.

        Returns:
            {"correctness": float, "quality": float, "reasoning": str}
        """
        raise NotImplementedError

    def score_batch(self, items: list[dict]) -> list[dict]:
        """Score a batch of items.

        Each item should contain keys ``prompt``, ``response`` and
        optionally ``thinking``. Returns one result dict per item.
        """
        results: list[dict] = []
        for item in items:
            prompt = item.get("prompt", "")
            response = item.get("response", "")
            thinking = item.get("thinking", "") or _extract_thinking(response)
            try:
                results.append(self.score(prompt, response, thinking))
            except Exception as exc:  # noqa: BLE001
                log.warning("Judge failed on item: %s", exc)
                results.append(_fallback_result(str(exc)))
        return results

    def unload(self) -> None:
        """Release any held resources (VRAM, connections)."""
        pass


# ── Parsing helpers ──────────────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
_SCORE_KEYS = ("correctness", "quality")


def _parse_judge_output(text: str) -> dict:
    """Parse the judge's JSON output into a result dict.

    Tolerant of surrounding prose / markdown fences.
    """
    # Strip markdown code fences if present.
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    # Try direct JSON parse first.
    data: Optional[dict] = None
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fall back to the first {...} block.
        m = _JSON_RE.search(cleaned)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = None

    if not isinstance(data, dict):
        return _fallback_result("unparseable judge output")

    correctness = _clamp_float(data.get("correctness"), default=0.0)
    quality = _clamp_float(data.get("quality"), default=0.0)
    reasoning = str(data.get("reasoning", "")).strip()
    return {
        "correctness": correctness,
        "quality": quality,
        "reasoning": reasoning,
    }


def _clamp_float(value: Any, default: float = 0.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, v))


def _fallback_result(reason: str = "") -> dict:
    return {"correctness": 0.0, "quality": 0.0, "reasoning": f"judge error: {reason}"}


def _extract_thinking(response: str) -> str:
    """Extract the content of the first thinking block from a response."""
    if THINK_TOKEN not in response:
        return ""
    if END_THINK_TOKEN in response:
        start = response.find(THINK_TOKEN) + len(THINK_TOKEN)
        end = response.find(END_THINK_TOKEN, start)
        if end > start:
            return response[start:end]
    start = response.find(THINK_TOKEN) + len(THINK_TOKEN)
    return response[start:]


def _extract_answer_text(response: str) -> str:
    """Extract the final-answer portion (text after the thinking block)."""
    if END_THINK_TOKEN in response:
        after = response.rsplit(END_THINK_TOKEN, 1)[1].strip()
        return after
    # No thinking block: the whole response is the "answer".
    return response.strip()


# ── Local judge (HF transformers) ────────────────────────────────────────────

class LocalJudge(BaseJudge):
    """LLM-as-judge using a local HuggingFace model.

    Can use the same checkpoint as the policy (Self-Rewarding LM style)
    or a separate, stronger local model. The model is loaded lazily on
    first use so importing this module is cheap.
    """

    def __init__(
        self,
        model_path: str = MODEL_ID,
        dtype: str = MODEL_DTYPE,
        device: str = "cuda",
        max_new_tokens: int = 512,
    ):
        self.model_path = model_path
        self.dtype = dtype
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None
        self._loaded = False

    def _load(self) -> None:
        """Lazy-load the model and tokenizer onto the GPU."""
        if self._loaded:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
            "fp32": torch.float32,
        }
        _dtype = dtype_map.get(self.dtype, torch.bfloat16)

        log.info("Loading judge model: %s (%s)", self.model_path, self.dtype)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            dtype=_dtype,
            device_map=self.device,
            trust_remote_code=True,
        )
        self._model.eval()
        self._loaded = True

    def _generate(self, prompt: str) -> str:
        """Run a single generation and return decoded text."""
        import torch

        self._load()
        assert self._tokenizer is not None and self._model is not None

        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_p=1.0,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(new_tokens, skip_special_tokens=True)

    def score(self, prompt: str, response: str, thinking: str = "") -> dict:
        thinking = thinking or _extract_thinking(response)
        answer = _extract_answer_text(response)
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            prompt=prompt,
            thinking=thinking or "(no thinking trace)",
            answer=answer or "(no explicit answer)",
        )
        try:
            raw = self._generate(judge_prompt)
        except Exception as exc:  # noqa: BLE001
            log.warning("LocalJudge generation failed: %s", exc)
            return _fallback_result(str(exc))
        return _parse_judge_output(raw)

    def unload(self) -> None:
        """Free VRAM by deleting the model and clearing caches."""
        import gc

        import torch

        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log.info("LocalJudge unloaded.")


# ── API judge (OpenAI / Anthropic) ───────────────────────────────────────────

class APIJudge(BaseJudge):
    """LLM-as-judge using an external API (OpenAI or Anthropic)."""

    def __init__(
        self,
        provider: str,
        api_key: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 512,
        temperature: float = 0.0,
        api_base: str = None,
    ):
        provider = (provider or "").lower().strip()
        if provider not in ("openai", "anthropic"):
            raise ValueError(f"Unsupported provider: {provider!r} (use 'openai' or 'anthropic')")
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.api_base = api_base
        self._client = None

    def _ensure_client(self) -> None:
        if self._client is not None:
            return
        if self.provider == "openai":
            from openai import OpenAI

            if self.api_base:
                self._client = OpenAI(api_key=self.api_key, base_url=self.api_base)
            else:
                self._client = OpenAI(api_key=self.api_key)
        else:  # anthropic
            from anthropic import Anthropic

            self._client = Anthropic(api_key=self.api_key)

    def _call_api(self, prompt: str) -> str:
        self._ensure_client()
        if self.provider == "openai":
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a strict JSON-only judge."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            return resp.choices[0].message.content or ""
        else:  # anthropic
            resp = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are a strict JSON-only judge.",
                messages=[{"role": "user", "content": prompt}],
            )
            # Anthropic returns a list of content blocks.
            parts = [getattr(b, "text", "") for b in resp.content]
            return "".join(parts)

    def score(self, prompt: str, response: str, thinking: str = "") -> dict:
        thinking = thinking or _extract_thinking(response)
        answer = _extract_answer_text(response)
        judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
            prompt=prompt,
            thinking=thinking or "(no thinking trace)",
            answer=answer or "(no explicit answer)",
        )
        try:
            raw = self._call_api(judge_prompt)
        except Exception as exc:  # noqa: BLE001
            log.warning("APIJudge call failed: %s", exc)
            return _fallback_result(str(exc))
        return _parse_judge_output(raw)

    def unload(self) -> None:
        self._client = None


# ── Rule-based judge ─────────────────────────────────────────────────────────

_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_BOXED_RE = re.compile(r"\\boxed\{([^{}]*)\}")
_ANSWER_IS_RE = re.compile(
    r"(?:final\s*answer|the\s*answer\s*is|answer\s*[:=])\s*\**\s*([^\n]+)",
    re.IGNORECASE,
)


class RuleBasedJudge(BaseJudge):
    """Judge for prompts with verifiable answers (math, code).

    No model is loaded. Correctness is determined by comparing the
    extracted answer against an expected answer (when provided) or by
    syntactic validity (for code). Quality is a simple heuristic based
    on whether a reasoning trace is present and well-formed.
    """

    def __init__(
        self,
        expected_answers: Optional[dict[str, str]] = None,
        task_type: str = "auto",
    ):
        # expected_answers: mapping prompt -> expected answer string.
        self.expected_answers = expected_answers or {}
        self.task_type = task_type  # "auto" | "math" | "code"

    def score(self, prompt: str, response: str, thinking: str = "") -> dict:
        thinking = thinking or _extract_thinking(response)
        answer_text = _extract_answer_text(response)
        task_type = self._detect_task_type(prompt, response)

        if task_type == "code":
            correctness, reasoning = self._score_code(response)
        else:
            correctness, reasoning = self._score_math(prompt, answer_text)

        quality = self._heuristic_quality(thinking, answer_text, correctness)
        return {
            "correctness": correctness,
            "quality": quality,
            "reasoning": reasoning,
        }

    def _detect_task_type(self, prompt: str, response: str) -> str:
        if self.task_type != "auto":
            return self.task_type
        if "```" in response or re.search(r"\b(def|class|import|print)\b", response):
            return "code"
        return "math"

    def _score_math(self, prompt: str, answer_text: str) -> tuple[float, str]:
        expected = self.expected_answers.get(prompt)
        extracted = _extract_math_answer(answer_text)
        if expected is None:
            # No ground truth: cannot verify correctness deterministically.
            return 0.0, "no expected answer provided; cannot verify"
        expected_num = _first_number(expected)
        extracted_num = _first_number(extracted)
        if expected_num is not None and extracted_num is not None:
            if abs(expected_num - extracted_num) < 1e-6:
                return 1.0, "numeric answer matches expected"
            return 0.0, f"expected {expected_num}, got {extracted_num}"
        if extracted.strip().lower() == expected.strip().lower():
            return 1.0, "exact text match"
        return 0.0, "answer does not match expected"

    def _score_code(self, response: str) -> tuple[float, str]:
        blocks = _CODE_BLOCK_RE.findall(response)
        if not blocks:
            return 0.0, "no code block found"
        ok = 0
        errors: list[str] = []
        for block in blocks:
            try:
                ast.parse(block)
                ok += 1
            except SyntaxError as exc:
                errors.append(f"SyntaxError: {exc.msg}")
        if ok == len(blocks):
            return 1.0, f"all {ok} code block(s) parse successfully"
        if ok > 0:
            return 0.5, f"{ok}/{len(blocks)} code blocks parse; errors: {errors[0]}"
        return 0.0, f"code does not parse: {errors[0] if errors else 'unknown'}"

    @staticmethod
    def _heuristic_quality(
        thinking: str, answer: str, correctness: float
    ) -> float:
        """Simple quality heuristic in [0, 1].

        - A present, non-trivial thinking trace contributes up to 0.5.
        - A present, non-empty final answer contributes up to 0.3.
        - Correctness contributes 0.2.
        """
        score = 0.0
        if thinking and len(thinking.split()) >= 5:
            score += min(0.5, len(thinking.split()) / 200.0 * 0.5)
        if answer and answer.strip():
            score += 0.3
        score += 0.2 * correctness
        return max(0.0, min(1.0, score))


def _extract_math_answer(text: str) -> str:
    """Extract a math answer from text, preferring boxed/answer patterns."""
    m = _BOXED_RE.search(text)
    if m:
        return m.group(1).strip()
    m = _ANSWER_IS_RE.search(text)
    if m:
        return m.group(1).strip()
    return text.strip()


def _first_number(s: str) -> Optional[float]:
    s = s.replace(",", "")
    m = _NUMBER_RE.search(s)
    return float(m.group(0)) if m else None


# ── Factory ──────────────────────────────────────────────────────────────────

def get_judge(mode: str, **kwargs: Any) -> BaseJudge:
    """Factory for judge instances.

    Args:
        mode: "local" | "api" | "rule"
        **kwargs: Backend-specific arguments.

    Returns:
        A BaseJudge instance.
    """
    mode = (mode or "").lower().strip()
    if mode == "local":
        return LocalJudge(**kwargs)
    if mode == "api":
        return APIJudge(**kwargs)
    if mode == "rule":
        return RuleBasedJudge(**kwargs)
    raise ValueError(f"Unknown judge mode: {mode!r} (use 'local', 'api', or 'rule')")
