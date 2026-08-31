"""
lazyRLGYM — Inference / Rollout Generation (transformers local)
================================================================
Generates N completions per prompt for RL rollouts by loading the
model locally with HuggingFace ``transformers``.

Design:
  * The model is loaded onto the GPU via ``AutoModelForCausalLM`` and
    registered with the central VRAM tracker so it can be unloaded
    before training starts.
  * ``generate`` produces ``n_samples`` completions for a single prompt
    using ``model.generate(do_sample=True, ...)``.
  * ``generate_batch`` iterates over prompts in mini-batches, reducing
    the batch size on OOM and retrying automatically.
  *  thinking blocks are parsed from each response.
"""
from __future__ import annotations

import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import THINK_TOKEN, END_THINK_TOKEN, MODEL_DTYPE, DEVICE, MAX_SEQ_LEN
from utils.vram import register, unload


# Regex to extract the first  ...  block (non-greedy, DOTALL).
_THINK_RE = re.compile(
    re.escape(THINK_TOKEN) + r"(.*?)" + re.escape(END_THINK_TOKEN),
    re.DOTALL,
)


def parse_thinking(text: str) -> tuple[str, str]:
    """Split generated text into (thinking_text, answer_text).

    The DeepSeek-R1 format wraps reasoning in  ... .
    Anything after the closing tag is the final answer. If no thinking
    block is present, the entire text is treated as the answer.
    """
    match = _THINK_RE.search(text)
    if match is None:
        # No explicit thinking block — everything is the answer.
        return "", text.strip()

    thinking_text = match.group(1).strip()
    # Answer is whatever follows the closing  tag.
    end = match.end()
    answer_text = text[end:].strip()
    return thinking_text, answer_text


# Map config dtype strings to torch dtypes.
_DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float16": torch.float16,
    "fp16": torch.float16,
    "float32": torch.float32,
    "fp32": torch.float32,
}


class RolloutGenerator:
    """Generates RL rollouts (N completions per prompt) via a local model.

    The model is loaded directly onto the GPU with ``transformers`` and
    registered with the VRAM manager so it can be freed before training.
    """

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
    ):
        self.model_path = model_path
        self.dtype_str = dtype
        self.torch_dtype = _DTYPE_MAP.get(dtype, torch.bfloat16)

        print(f"[Rollout] Loading model from {model_path} (dtype={dtype}) ...")
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

        # Get think/end-think token IDs for proper thinking block extraction.
        # The R1 chat template ends with <think>, so the generated tokens
        # contain the thinking content + </think> + answer. We split on
        # the </think> token ID to separate them reliably.
        self.end_think_token_id = self.tokenizer.convert_tokens_to_ids(END_THINK_TOKEN)
        if self.end_think_token_id == self.tokenizer.unk_token_id:
            print(f"[Rollout] WARNING: '{END_THINK_TOKEN}' not in tokenizer vocabulary")
            self.end_think_token_id = None

        register("policy", self.model)
        print("[Rollout] Model loaded and ready.")

    # ── helpers ─────────────────────────────────────────────────────────────

    def _build_result_from_ids(self, new_ids: torch.Tensor) -> dict:
        """Build a result dict from generated token IDs.

        Splits the generated tokens at the  token to separate
        thinking content from the final answer. This is more reliable than
        text-based parsing because skip_special_tokens strips the think
        markers from decoded text.
        """
        # Convert to 1D tensor if needed
        if new_ids.dim() > 1:
            new_ids = new_ids.squeeze(0)

        if self.end_think_token_id is not None:
            # Find the first  token in the generated sequence
            mask = (new_ids == self.end_think_token_id)
            end_think_positions = mask.nonzero(as_tuple=True)[0]

            if len(end_think_positions) > 0:
                end_pos = end_think_positions[0].item()
                thinking_ids = new_ids[:end_pos]
                answer_ids = new_ids[end_pos + 1:]
            else:
                # No  found — model is still thinking (truncated by max_tokens).
                # Treat everything as thinking content.
                thinking_ids = new_ids
                answer_ids = new_ids[:0]  # empty tensor
        else:
            # Fallback: text-based parsing
            full_text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            thinking_text, answer_text = parse_thinking(full_text)
            return {
                "text": full_text,
                "thinking_text": thinking_text,
                "answer_text": answer_text,
                "thinking_tokens": len(thinking_text.split()) if thinking_text else 0,
                "answer_tokens": len(answer_text.split()) if answer_text else 0,
                "total_tokens": new_ids.shape[0],
            }

        # Decode each part separately with skip_special_tokens=True
        thinking_text = self.tokenizer.decode(thinking_ids, skip_special_tokens=True).strip()
        answer_text = self.tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
        full_text = self.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        return {
            "text": full_text,
            "thinking_text": thinking_text,
            "answer_text": answer_text,
            "thinking_tokens": thinking_ids.shape[0],
            "answer_tokens": answer_ids.shape[0],
            "total_tokens": new_ids.shape[0],
        }

    def _generate_tokens(
        self,
        prompt: str,
        n_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> tuple[torch.Tensor, int]:
        """Tokenize a single prompt and generate n_samples completions.

        Returns (output_ids, num_generated_tokens) where output_ids has
        shape (n_samples, prompt_len + gen_len).
        """
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
                num_return_sequences=n_samples,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        gen_len = output_ids.shape[1] - input_len
        return output_ids, gen_len

    # ── public API ──────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        n_samples: int = 1,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> list[dict]:
        """Generate n_samples completions for a single prompt.

        Returns a list of result dicts (one per sample).
        """
        output_ids, gen_len = self._generate_tokens(
            prompt, n_samples, max_new_tokens, temperature, top_p
        )
        input_len = output_ids.shape[1] - gen_len

        results: list[dict] = []
        for i in range(output_ids.shape[0]):
            new_ids = output_ids[i, input_len:]
            results.append(self._build_result_from_ids(new_ids))
        return results

    def generate_batch(
        self,
        prompts: list[str],
        n_samples: int = 8,
        max_new_tokens: int = 4096,
        temperature: float = 0.7,
        top_p: float = 0.95,
        batch_size: int = 4,
    ) -> list[list[dict]]:
        """Generate for multiple prompts in mini-batches.

        Prompts are processed ``batch_size`` at a time.  If a batch
        triggers an OOM, the batch size is halved and the batch is
        retried until it succeeds (or batch_size drops to 1).

        Returns a list aligned with ``prompts``; each element is the list
        of n_samples completion dicts for that prompt.
        """
        all_results: list[list[dict]] = [None] * len(prompts)  # type: ignore[list-item]

        idx = 0
        while idx < len(prompts):
            current_bs = min(batch_size, len(prompts) - idx)
            batch_prompts = prompts[idx : idx + current_bs]

            success = False
            while current_bs >= 1 and not success:
                try:
                    batch_results = self._generate_batch_inner(
                        batch_prompts[:current_bs],
                        n_samples=n_samples,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    success = True
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    current_bs = max(1, current_bs // 2)
                    if current_bs == 1 and len(batch_prompts) > 1:
                        # Retry one-by-one.
                        batch_prompts = batch_prompts[:current_bs]
                    elif current_bs < 1:
                        raise
                    print(
                        f"[Rollout] OOM — reduced batch_size to {current_bs}, retrying ..."
                    )

            for j, res in enumerate(batch_results):
                all_results[idx + j] = res
            idx += len(batch_results)

        return all_results  # type: ignore[return-value]

    def _generate_batch_inner(
        self,
        prompts: list[str],
        n_samples: int,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> list[list[dict]]:
        """Generate n_samples completions for each prompt in a batch.

        Uses left-padded batching so all prompts in the batch are
        processed in a single ``model.generate`` call.
        """
        # Build chat-templated inputs.
        input_texts = []
        for prompt in prompts:
            messages = [{"role": "user", "content": prompt}]
            input_texts.append(
                self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        inputs = self.tokenizer(
            input_texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
            padding=True,
        ).to(DEVICE)

        input_len = inputs["input_ids"].shape[1]
        total_seqs = len(prompts) * n_samples

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=max(temperature, 1e-4),
                top_p=top_p,
                num_return_sequences=n_samples,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_len = output_ids.shape[1] - input_len

        batch_results: list[list[dict]] = []
        for i in range(len(prompts)):
            sample_results: list[dict] = []
            for s in range(n_samples):
                seq_idx = i * n_samples + s
                new_ids = output_ids[seq_idx, input_len:]
                sample_results.append(self._build_result_from_ids(new_ids))
            batch_results.append(sample_results)
        return batch_results

    def unload(self) -> None:
        """Free the model from VRAM."""
        unload("policy")
        self.model = None  # type: ignore[assignment]
        self.tokenizer = None  # type: ignore[assignment]
        print("[Rollout] Model unloaded.")
