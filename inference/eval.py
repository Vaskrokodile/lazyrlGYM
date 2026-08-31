"""
lazyRLGYM — Inference / Evaluation Utilities (transformers local)
==================================================================
Evaluation metrics for the outer loop's safety gates:
  * Held-out win-rate / mean score / mean length
  * pass@K
  * Self-BLEU (diversity)

Loads the model locally with HuggingFace ``transformers`` and registers
it with the central VRAM tracker so it can be freed after evaluation.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Callable

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import THINK_TOKEN, END_THINK_TOKEN, MODEL_DTYPE, DEVICE, MAX_SEQ_LEN
from utils.vram import register, unload
from inference.rollout import parse_thinking


# ── Self-BLEU (no external deps) ────────────────────────────────────────────

def _ngrams(tokens: list[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def _modified_precision(candidate: list[str], references: list[list[str]], n: int) -> tuple[int, int]:
    """Return (clipped_match_count, total_ngram_count) for one candidate
    against a set of references (used as the 'corpus' for self-BLEU)."""
    cand_ngrams = _ngrams(candidate, n)
    if not cand_ngrams:
        return 0, 0

    # Maximum reference counts (across all other responses).
    max_ref: Counter = Counter()
    for ref in references:
        ref_ngrams = _ngrams(ref, n)
        for ng, c in ref_ngrams.items():
            if c > max_ref[ng]:
                max_ref[ng] = c

    clipped = 0
    total = 0
    for ng, c in cand_ngrams.items():
        clipped += min(c, max_ref.get(ng, 0))
        total += c
    return clipped, total


def _brevity_penalty(cand_len: int, ref_len: int) -> float:
    if cand_len > ref_len:
        return 1.0
    if cand_len == 0:
        return 0.0
    return math.exp(1.0 - ref_len / cand_len)


def self_bleu(responses: list[str], max_n: int = 4) -> float:
    """Compute Self-BLEU across a list of responses.

    For each response, every *other* response is treated as a reference.
    Lower Self-BLEU => higher diversity. Returns the mean BLEU score.
    """
    tokenized = [r.split() for r in responses if r.strip()]
    if len(tokenized) < 2:
        return 0.0

    weights = [1.0 / max_n] * max_n
    scores: list[float] = []

    for i, cand in enumerate(tokenized):
        refs = [tok for j, tok in enumerate(tokenized) if j != i]
        if not refs:
            continue

        precisions: list[float] = []
        for n in range(1, max_n + 1):
            clipped, total = _modified_precision(cand, refs, n)
            if total == 0:
                precisions.append(0.0)
            else:
                precisions.append(clipped / total)

        # Geometric mean of precisions (with smoothing for zeros).
        if any(p == 0.0 for p in precisions):
            # Smooth: add a tiny epsilon to avoid log(0).
            log_sum = sum(math.log(max(p, 1e-10)) for p in precisions)
            geo = math.exp(log_sum / len(precisions))
        else:
            log_sum = sum(math.log(p) for p in precisions)
            geo = math.exp(log_sum / len(precisions))

        ref_len = max(len(r) for r in refs)
        bp = _brevity_penalty(len(cand), ref_len)
        scores.append(bp * geo)

    return sum(scores) / len(scores) if scores else 0.0


# ── dtype mapping ───────────────────────────────────────────────────────────

_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


# ── Evaluator ───────────────────────────────────────────────────────────────

class Evaluator:
    """Runs held-out evaluation, pass@K, and diversity metrics locally."""

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
    ):
        self.model_path = model_path
        self.dtype_str = dtype
        self.torch_dtype = _DTYPE_MAP.get(dtype, torch.bfloat16)

        print(f"[Eval] Loading model from {model_path} (dtype={dtype}) ...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            padding_side="left",
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=self.torch_dtype,
            attn_implementation="sdpa",
        )
        self.model.to(DEVICE)
        self.model.eval()

        # Get end-think token ID for proper thinking/answer splitting
        self.end_think_token_id = self.tokenizer.convert_tokens_to_ids(END_THINK_TOKEN)
        if self.end_think_token_id == self.tokenizer.unk_token_id:
            self.end_think_token_id = None

        register("eval", self.model)
        print("[Eval] Model loaded and ready.")

    # ── helpers ─────────────────────────────────────────────────────────────

    def _decode_with_thinking(self, new_ids: torch.Tensor) -> str:
        """Decode generated tokens, preserving  markers for parse_thinking.

        If we have the end_think token ID, we decode the answer portion
        (after ) separately and reconstruct the text with markers
        so parse_thinking can split it correctly.
        """
        if self.end_think_token_id is not None:
            mask = (new_ids == self.end_think_token_id)
            positions = mask.nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                pos = positions[0].item()
                thinking_text = self.tokenizer.decode(new_ids[:pos], skip_special_tokens=True).strip()
                answer_text = self.tokenizer.decode(new_ids[pos + 1:], skip_special_tokens=True).strip()
                return f"{THINK_TOKEN}\n{thinking_text}\n{END_THINK_TOKEN}\n{answer_text}"
            else:
                # No end_think — all thinking (truncated)
                thinking_text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()
                return f"{THINK_TOKEN}\n{thinking_text}\n{END_THINK_TOKEN}\n"
        # Fallback: just decode normally
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)

    def _generate_one(
        self,
        prompt: str,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        """Generate a single response string for a prompt."""
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(DEVICE)

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 1e-4),
                top_p=top_p,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        input_len = inputs["input_ids"].shape[1]
        new_ids = output_ids[0, input_len:]
        return self._decode_with_thinking(new_ids)

    def _generate_k(
        self,
        prompt: str,
        k: int,
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> list[str]:
        """Generate k samples for a prompt in a single generate call."""
        messages = [{"role": "user", "content": prompt}]
        input_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(DEVICE)

        input_len = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 1e-4),
                top_p=top_p,
                num_return_sequences=k,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        results: list[str] = []
        for i in range(output_ids.shape[0]):
            new_ids = output_ids[i, input_len:]
            results.append(self._decode_with_thinking(new_ids))
        return results

    @staticmethod
    def _extract_prompt(item: dict | str) -> str:
        if isinstance(item, str):
            return item
        return item.get("prompt", item.get("text", ""))

    # ── public metrics ──────────────────────────────────────────────────────

    def evaluate_heldout(
        self,
        prompts: list[dict],
        judge_fn: Callable[[str, str], float],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> dict:
        """Generate 1 response per held-out prompt, score with judge_fn.

        judge_fn(prompt, response) -> float score in [0, 1].
        Returns aggregate metrics plus per-prompt results.
        """
        items = []
        for item in prompts:
            prompt = self._extract_prompt(item)
            if prompt:
                items.append((item, prompt))

        results: list[dict] = []
        scores: list[float] = []
        lengths: list[int] = []

        for (item, prompt) in items:
            try:
                response = self._generate_one(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print("[Eval] OOM during heldout eval — retrying single prompt ...")
                response = self._generate_one(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )

            _, answer_text = parse_thinking(response)
            score = float(judge_fn(prompt, answer_text))
            length = len(answer_text.split())

            scores.append(score)
            lengths.append(length)
            results.append(
                {
                    "prompt": prompt,
                    "response": response,
                    "answer": answer_text,
                    "score": score,
                    "length": length,
                }
            )

        n = len(scores) or 1
        mean_score = sum(scores) / n if scores else 0.0
        winrate = sum(1 for s in scores if s > 0.5) / n if scores else 0.0
        mean_length = sum(lengths) / n if lengths else 0.0

        return {
            "winrate": winrate,
            "mean_score": mean_score,
            "mean_length": mean_length,
            "results": results,
        }

    def compute_pass_at_k(
        self,
        prompts: list[dict],
        k: int,
        judge_fn: Callable[[str, str], float],
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> float:
        """For each prompt, generate k samples; a prompt 'passes' if any
        sample scores > 0.5.  Returns the fraction of prompts that pass."""
        if not prompts:
            return 0.0

        items = []
        for item in prompts:
            prompt = self._extract_prompt(item)
            if prompt:
                items.append(prompt)

        passed = 0
        for prompt in items:
            try:
                samples = self._generate_k(
                    prompt,
                    k,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"[Eval] OOM on pass@k (k={k}) — retrying one-by-one ...")
                samples = []
                for _ in range(k):
                    samples.append(
                        self._generate_one(
                            prompt,
                            max_new_tokens=max_new_tokens,
                            temperature=temperature,
                            top_p=top_p,
                        )
                    )

            for resp in samples:
                _, answer_text = parse_thinking(resp)
                if judge_fn(prompt, answer_text) > 0.5:
                    passed += 1
                    break

        return passed / len(prompts)

    def compute_self_bleu(self, responses: list[str]) -> float:
        """Compute Self-BLEU diversity metric (lower = more diverse)."""
        return self_bleu(responses)

    def unload(self) -> None:
        """Free the model from VRAM."""
        unload("eval")
        self.model = None  # type: ignore[assignment]
        self.tokenizer = None  # type: ignore[assignment]
        print("[Eval] Model unloaded.")
