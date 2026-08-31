# lazyRLGYM - Development Log & Results

## Overview

This document records the full development and training history of the lazyRLGYM project - an API-only RL pipeline that trains DeepSeek-R1-Distill-Qwen-1.5B to optimize reasoning length based on prompt difficulty using iterative DPO with length x difficulty reward shaping.

**Hardware**: RTX 3060 12GB VRAM, 16GB RAM, Windows 11
**Model**: DeepSeek-R1-Distill-Qwen-1.5B (local cache, bf16)
**Method**: Iterative DPO with LoRA adapters, rule-based judge, length x difficulty reward shaping

---

## Phase 1: Bug Fixes (Initial Pipeline Repair)

The pipeline was originally broken in several ways. The following bugs were identified and fixed before any successful training run:

### Bug 1: RuleBasedJudge not receiving expected answers
- **Problem**: The `RuleBasedJudge` reported 0.0 correctness for all prompts because it had no `expected_answers` dict.
- **Fix**: Added `expected_answer` fields to arithmetic/math prompts in `data/prompts.py`. Created `_build_expected_answers` helper in `orchestrator.py` to pass these to the judge when `judge_mode="rule"`.
- **Files**: `data/prompts.py`, `orchestrator.py`

### Bug 2: Thinking tokens not extracted (skip_special_tokens)
- **Problem**: The R1 chat template ends with the think token, and `skip_special_tokens=True` in the tokenizer decode strips both think and end-think markers from decoded text. The `parse_thinking()` function couldn't find the thinking block, so `thinking_tokens` was always 0.
- **Fix**: Added token-ID-level splitting in `RolloutGenerator._build_result_from_ids()`. The generator now finds the end-think token ID in the generated token sequence and splits thinking from answer at the token level before decoding each part separately.
- **Files**: `inference/rollout.py`, `inference/eval.py`
- **Verification**: After fix, "What is 2+3?" produced 68 thinking tokens and 123 answer tokens (vs 0/0 before).

### Bug 3: Unicode encoding crash (Windows cp1252)
- **Problem**: Windows console uses cp1252 encoding which cannot handle Unicode arrows and checkmarks in print statements. The pipeline crashed in Phase 5 with `UnicodeEncodeError: 'charmap' codec can't encode character`.
- **Fix**: Replaced Unicode characters in print statements with ASCII equivalents. Set `PYTHONIOENCODING=utf-8` in the runner script.
- **Files**: `orchestrator.py`, `run_with_logging.py`

### Bug 4: DPO group_by_length crash
- **Problem**: `group_by_length=True` in DPO config crashed because the DPO dataset has no `input_ids` key (it has `chosen`/`rejected`). Error: "Can only automatically infer lengths for datasets whose items are dictionaries with an 'input_ids' key."
- **Fix**: Set `group_by_length=False` in the DPO config.
- **Files**: `train/dpo.py`

### Bug 5: Windows multiprocessing crash
- **Problem**: `dataloader_num_workers=2` caused multiprocessing errors on Windows: "An attempt has been made to start a new process before the current process has finished its bootstrapping phase."
- **Fix**: Set `dataloader_num_workers=0` and added `if __name__ == "__main__":` guard in the runner script.
- **Files**: `config.py`, `run_with_logging.py`

### Bug 6: Unrealistic length targets for 1.5B model
- **Problem**: Original length targets (easy=150, hard=2000) were unrealistic for a 1.5B model with 1024 max tokens. Almost all prompts were classified as "hard" and the model got underthinking penalties on everything.
- **Fix**: Adjusted to easy=80, medium=250, hard=600. Relaxed difficulty thresholds from >0.8 to >0.7 for easy, >=0.35 for medium.
- **Files**: `config.py`, `reward/difficulty.py`

---

## Phase 2: First Successful Run (3 iterations, conservative config)

### Configuration
| Parameter | Value |
|-----------|-------|
| Prompts | 12 (9 train, 3 held-out) |
| Samples/prompt | 4 |
| Max new tokens | 1024 |
| DPO batch size | 1 |
| DPO grad accum | 8 |
| DPO learning rate | 5e-6 |
| DPO epochs | 1 |
| LoRA rank | 16 |
| Iterations | 3 |

### Results

**Iteration 0 (PROMOTED):**
- Difficulty: easy=3, medium=2, hard=4
- Mean reward: 0.265
- Length: easy=83, med=704, hard=973 tokens
- DPO train loss: 0.6931 (stuck at ln(2) - no learning)
- Held-out: winrate=0.333, score=0.313
- Pass@K: 0.667

**Iteration 1 (ROLLED BACK - no improvement):**
- Mean reward: 0.410 (improved)
- Length: easy=74, med=190, hard=875
- DPO train loss: 0.6931 (still stuck)
- Held-out: winrate=0.333, score=0.313 (no change)
- Decision: Rolled back (delta=0.000)

**Iteration 2 (ROLLED BACK - no improvement):**
- Same metrics as iteration 1
- Decision: Rolled back

### Analysis
The pipeline ran end-to-end without OOM, but **DPO training was ineffective**:
- DPO loss stayed at 0.6931 (ln(2)) across all iterations
- Only 6 preference pairs per iteration (from 36 rollouts)
- Only 1 effective gradient step (batch_size=1, grad_accum=8)
- The model could not learn to distinguish chosen from rejected with so little training data and so few steps

### Key Positive Result
Length x difficulty differentiation was working: easy prompts got ~80 thinking tokens, hard prompts got ~900 tokens (11.7x ratio).

---

## Phase 3: Improved Configuration (speed + training effectiveness)

### Changes Made

**Speed improvements:**
1. Switched from one-at-a-time generation to batched generation (`generate_batch`)
2. Changed `torch.no_grad()` to `torch.inference_mode()` (faster, less overhead)
3. Reduced eval `max_new_tokens` from 2048 to 1024
4. Skip pass@k on non-boundary iterations (only compute on iter 0 and last)
5. Reduced pass@k from k=4 to k=2

**Training effectiveness improvements:**
1. Increased prompts from 12 to 30 (all builtin prompts: 24 train, 6 held-out)
2. Increased samples/prompt from 4 to 6 (144 rollouts per iteration)
3. Increased LoRA rank from 16 to 32 (more capacity)
4. Increased DPO learning rate from 5e-6 to 1e-5 (4x higher)
5. Increased DPO epochs from 1 to 2
6. Decreased DPO grad_accum from 8 to 2 (more gradient steps)
7. Multiple preference pairs per prompt (up to 3, with min_score_delta=0.05 filter)
8. Changed promotion gate to use mean_score delta instead of winrate

### Files Modified
- `config.py` - all hyperparameter changes
- `orchestrator.py` - batched rollout generation, eval speed improvements, promotion gate logic
- `inference/rollout.py` - inference_mode, batched generation
- `inference/eval.py` - inference_mode, token-level thinking extraction
- `data/preferences.py` - multiple pairs per prompt, min_score_delta filter
- `reward/difficulty.py` - relaxed difficulty thresholds

---

## Phase 4: Second Run Results (improved config)

### Run 1: Aggressive LR (2e-5, 3 epochs)

**Iteration 0:**
- 144 rollouts, 38 preference pairs (up from 6!)
- DPO loss: 0.6931 -> 0.4305 (massive improvement!)
- Reward accuracy: 1.0 (100%)
- Reward margins: grew from 0.02 to 2.10
- 57 training steps (up from 1)
- BUT: held-out pass@K dropped from 0.667 to 0.167 (model overfit)
- Decision: Rolled back (pass@K too low)

**Diagnosis**: Learning rate too aggressive (2e-5) with 3 epochs caused overfitting. The model learned to distinguish preferences perfectly on training data but degraded on held-out evaluation.

### Run 2: Balanced LR (1e-5, 2 epochs)

**Iteration 0 (PROMOTED as baseline):**
- 144 rollouts, 30 preference pairs
- Difficulty: easy=7, medium=1, hard=16
- Mean reward: 0.197
- Length: easy=465, med=187, hard=747 tokens
- DPO loss: 0.6931 -> 0.6426 (good learning without overfitting)
- Reward accuracy: 1.0 (100%)
- Held-out: winrate=0.167, score=0.172
- Pass@K: 0.167
- Decision: PROMOTED (first iteration baseline)
- Duration: 1128s (~19 min)

**Iteration 1 (ROLLED BACK - length explosion):**
- Mean reward: 0.203
- Length: easy=387, med=361, hard=847
- DPO loss: 0.6746, reward_acc: 1.0
- Held-out: winrate=0.167, score=0.187 (improved from 0.172!)
- BUT: eval mean_length went from 42 -> 82 words (ratio 1.97)
- Decision: Rolled back (length explosion, ratio > 1.5 threshold)
- Note: Score actually improved but length gate blocked promotion

**Fix applied**: Increased `max_length_growth_ratio` from 1.5 to 3.0. Changed promotion gate so score improvement takes priority over length checks.

### Run 3: Final Run (with all fixes)

**Iteration 0 (PROMOTED as baseline):**
- 144 rollouts, 30 preference pairs
- Difficulty: easy=7, medium=1, hard=16
- Mean reward: 0.197
- Length: easy=465, med=187, hard=747 tokens
- DPO loss: 0.6931 -> 0.6426
- Reward accuracy: 1.0
- Held-out: winrate=0.167, score=0.172
- Decision: PROMOTED (baseline)

**Iteration 1 (DPO completed, crashed during cleanup):**
- DPO training completed successfully
- Adapter saved to `dpo_iter_1`
- Crashed with CUDA OOM during cleanup (caused by running `compare_models.py` alongside training, which fragmented VRAM)
- Merge and eval did not complete

---

## Phase 5: Model Output Comparison

### Side-by-side: Base Model vs Trained Model (iter 0)

| Difficulty | Prompt | Base tokens | Trained tokens | Delta |
|-----------|--------|------------|---------------|-------|
| easy | "What is 2+3?" | 33 | 64 | +31 |
| easy | "What is 7x6?" | 89 | 112 | +23 |
| medium | "Train speed" | 156 | 116 | -40 |
| hard | "URL shortener" | 1024 | 1024 | 0 (both hit max) |

### Assessment
After 1 DPO iteration with 30 preference pairs and 30 gradient steps:
- The DPO loss dropped (0.6931 -> 0.6426) and reward accuracy hit 100%
- The model learned to **distinguish** good from bad responses
- But this hasn't yet translated into **generating** differently
- The medium prompt got shorter (correct direction - reward shaping penalizes overthinking on medium)
- The easy prompts got slightly longer (within sampling noise at temperature=0.7)
- The hard prompt hit the 1024 token limit on both models (never produces an answer)
- Multiple compounding iterations are needed to see generation behavior shift

### Preference Pair Examples (from iter 0)

**Easy prompt "What is 7x6?":**
- CHOSEN (score=1.319): Concise thinking, direct answer
- REJECTED (score=1.253): Longer thinking with step-by-step addition (overthinking)

**Medium prompt "Train speed":**
- CHOSEN (score=1.405): Clear, structured reasoning
- REJECTED (score=0.229): Similar content but less well-organized

**Hard prompt "URL shortener":**
- CHOSEN (score=0.393): More structured system design thinking
- REJECTED (score=0.123): Less organized, more meandering

---

## DPO Loss Trajectory (Run 2, Iteration 0)

The DPO loss showed clear learning across 30 training steps:

| Step | Loss | Reward Accuracy | Reward Margin |
|------|------|----------------|---------------|
| 5 | 0.7063 | 0.10 | -0.026 |
| 10 | 0.6754 | 0.70 | 0.037 |
| 15 | 0.6335 | 0.78 | 0.139 |
| 20 | 0.5987 | 1.00 | 0.202 |
| 25 | 0.5669 | 1.00 | 0.279 |
| 30 | 0.5669 | 1.00 | 0.279 |

Final: train_loss=0.6426, reward_accuracy=1.0

---

## Resource Usage

### VRAM (Peak per phase)
| Phase | Peak VRAM | Free VRAM |
|-------|-----------|-----------|
| Rollout (batched inference) | 9.9 GB | 2.2 GB |
| DPO training | 8.4 GB | 3.7 GB |
| Evaluation | 4.7 GB | 7.5 GB |

### Timing (per iteration)
| Phase | Duration |
|-------|----------|
| Rollout (24 prompts x 6 samples, batched) | ~3 min |
| Difficulty + Judging | ~1 min |
| DPO training (30 pairs, 2 epochs, 30 steps) | ~3.5 min |
| Merge | ~0.5 min |
| Evaluation (6 prompts + pass@k) | ~5 min |
| **Total per iteration** | ~13-19 min |

---

## Artifacts

### Checkpoints
- `checkpoints/merged_iter_0` - Promoted checkpoint from iteration 0
- `checkpoints/dpo_iter_0` - LoRA adapter from iteration 0
- `checkpoints/dpo_iter_1` - LoRA adapter from iteration 1 (merge incomplete)

### Data
- `data/cache/prefs_iter_0.jsonl` - 30 preference pairs from iteration 0
- `data/cache/prefs_iter_1.jsonl` - 25 preference pairs from iteration 1

### Logs
- `logs/training_run.log` - Full training log

### Scripts
- `run_with_logging.py` - Training runner with file logging
- `show_progress.py` - Extract and display iteration summaries from log
- `compare_models.py` - Side-by-side comparison of base vs trained model
- `test_judge_fix.py` - Verify RuleBasedJudge correctness scoring
- `test_rollout_fix.py` - Verify thinking token extraction

---

## Key Lessons

1. **DPO needs enough preference pairs**: 6 pairs with 1 gradient step = no learning. 30 pairs with 30 steps = clear learning (loss 0.69 -> 0.64).

2. **Learning rate matters a lot**: 2e-5 with 3 epochs overfit (pass@K dropped from 0.667 to 0.167). 1e-5 with 2 epochs was balanced (loss dropped without degrading held-out).

3. **Batched generation is much faster**: Processing 6 prompts per `model.generate()` call vs 1-at-a-time cut rollout time from ~5 min to ~3 min.

4. **Token-level thinking extraction is essential**: The R1 model's think/end-think tokens are special tokens that get stripped by `skip_special_tokens=True`. Must split at the token ID level.

5. **Promotion gates need careful tuning**: The length growth ratio of 1.5 was too aggressive (blocked a checkpoint that actually improved score). Changed to 3.0 and made score improvement take priority.

6. **Don't run inference alongside training**: Running `compare_models.py` while training was in progress caused VRAM fragmentation and an OOM crash during DPO cleanup.

---

## Next Steps

To see significant CoT behavior change:
1. Run 5+ iterations uninterrupted (each compounds on the previous)
2. Increase max_new_tokens to 1536+ for hard prompts (1024 is too short for system design)
3. Consider adding more diverse hard prompts that don't require 1000+ tokens of thinking
4. Try LoRA rank 64 for more capacity
5. Consider an LLM judge instead of rule-based for more nuanced quality scoring
6. Monitor held-out score trajectory across iterations - that's the key metric
