# lazyRLGYM — Research Brief

**Goal:** Train an LLM (via API only — no local model clone) to be *efficient* in its
reasoning: punish "too simple" thinking on hard prompts AND "too much" thinking on
easy prompts. The same model acts as judge. The agent has SSH access to a training
box to update its own policy.

**Date:** 2026-08-31
**Method:** 4 parallel research subagents (RLAIF, overthinking, API-only loops,
agentic SSH) synthesized below.

---

## TL;DR — What's possible and what's not

1. **API-only RL is feasible** via *iterative/batched* methods
   (Iterative DPO, Self-Rewarding LM, ReST^EM, RAFT). You do NOT need PPO/GRPO
   local gradients for the outer loop. The pattern is:
   `sample N → judge → build preference pairs → call fine-tuning API → redeploy`.

2. **The judge CAN be the same model as the policy** (Self-Rewarding LM, Yuan et al.
   2024). It works but has a documented self-preference bias — mitigate with a
   periodic external-judge audit.

3. **"Kill laziness / kill overthinking" reward shaping is an active 2025-2026
   research area.** The desired reward sign table (difficulty × reasoning length)
   is well-established and maps cleanly onto LASER-D / LEASH / Kimi k1.5 length
   rewards — but those papers all use *local gradients*. For API-only, you encode
   the same shaping as a *preference-construction rule* (short-correct > long-correct
   on easy prompts; long-correct > short-correct on hard prompts) and feed it to DPO.

4. **Difficulty estimation WITHOUT ground truth is possible via API**: sample
   consistency / self-certainty across N samples (arXiv:2402.13904, 2502.18581),
   or LLM-as-judge Bradley-Terry scoring (arXiv:2512.14220). This is the key enabler
   for your "is this prompt simple or complex?" signal.

5. **SSH self-modifying agent is novel — no public project does exactly this.**
   SEAL (arXiv:2506.10943) is the closest (model generates its own FT data +
   update directives). The critical safety constraint: **trust isolation** — the
   agent must NEVER have write access to its own eval suite, judge, or reward code.

---

## 1. The API-only training loop (recommended architecture)

```
┌─────────────────────────────────────────────────────────────┐
│  ORCHESTRATOR (you — not the LLM)                            │
│  prompt bank · iteration counter · budget · safety gates     │
└───────────┬─────────────────────────────────────────────────┘
            ▼
┌───────────────────────────┐
│ 1. ROLLOUT                 │  Call policy API N times per prompt
│    (API inference only)    │  Capture: prompt, reasoning trace, answer
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│ 2. DIFFICULTY ESTIMATE     │  Sample-consistency / self-certainty
│    (API, no ground truth)  │  → bucket prompt as easy/medium/hard
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│ 3. JUDGE (same model,      │  Score each (trace, answer) on:
│    judge-prompt mode)      │   - correctness (LLM-judge or verifier)
│                            │   - reasoning-length appropriateness
│                            │     given the difficulty bucket
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│ 4. PREFERENCE PAIRS        │  For each prompt, pick:
│    (construction rule)     │   chosen  = best score per difficulty
│                            │   rejected = worst score per difficulty
│                            │  Apply length-shaping rule (§3)
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│ 5. FINE-TUNE TRIGGER       │  Upload JSONL → provider FT API
│    (SSH to training box    │   OR SSH: run DPO/SFT job on cluster
│     OR provider FT API)    │  Poll until succeeded
└───────────┬───────────────┘
            ▼
┌───────────────────────────┐
│ 6. PROMOTION GATE          │  External controller runs held-out eval:
│    (NOT the LLM)           │   - score improved?
│                            │   - no reward hacking / length collapse?
│                            │   - safety evals pass?
│                            │  If yes → swap checkpoint in as new policy
│                            │  If no  → rollback
└───────────┬───────────────┘
            └──▶ iteration t+1
```

**Key design rules:**
- No gradients during rollouts — pure API inference.
- Judge = policy (same checkpoint, judge-prompt) — but audited periodically by an
  external stronger model on a hidden anchor set.
- The agent's SSH key can ONLY run a pre-approved training script and write to a
  controlled artifact path. It CANNOT reach the eval suite or judge code.
- The promotion gate is an external controller, never the LLM.

---

## 2. Method selection — what to actually use

| Your constraint | Recommended method | Why |
|---|---|---|
| API-only policy, no local weights | **Iterative DPO** (Self-Rewarding variant) | Outer loop = sample + judge + DPO job. No PPO/value model. |
| Cheapest possible pilot | **ReST^EM / RAFT** | Sample N, filter by reward, SFT on survivors. One FT call per iter. |
| Provider only supports SFT (no DPO) | **RAFT** (best-of-N + SFT on top-1) | No preference data needed. |
| Provider supports DPO | **Iterative DPO with length-shaped pairs** | Encodes your "kill laziness / kill overthinking" rule directly into chosen/rejected. |
| Want self-play, no external judge | **SPIN** | Model plays against prior checkpoint. Implicit preference. |
| Need Nash-equilibrium robustness | **SPPO** | Constant-sum game formulation. More iters, higher cost. |

**Avoid:** PPO, GRPO, d-RLAIF, Online DPO (true online) — all require local
gradients / on-policy backprop. Not API-compatible.

**Sweet spot for your idea:** Iterative DPO with a *length-and-difficulty-aware
preference construction rule*. This is the API-only analog of LASER-D / Kimi k1.5
length rewards.

---

## 3. The reward shaping — difficulty × reasoning length

This is the core of your "kill laziness / kill overthinking" idea. Assuming the
final answer is **correct** (wrong answers get negative reward regardless of length):

| Prompt difficulty | Too short (underthinking) | Optimal length | Too long (overthinking) |
|---|---|---|---|
| **Easy** | `0` (ok if correct) | `+` | `−` (the "2+3=?" case) |
| **Hard** | `−` (not enough reasoning) | `+` | `−` after the optimal peak |

**How to encode this in API-only DPO:**
For each prompt, after estimating difficulty `d` and collecting N traces of lengths
`L_1..L_N` with judge scores `s_1..s_N`:

```
effective_score_i = s_i  -  α(d) · penalty(L_i, d)

where:
  penalty(L, d) = max(0, L - L_target(d))        # overthinking
                + max(0, L_target(d) - L) · β(d) # underthinking (only penalize on hard)

  L_target(easy)  = small   (e.g. 50-150 tokens)
  L_target(hard)  = large   (e.g. 500-2000 tokens)
  α(easy)  > α(hard)        # punish overthinking more on easy prompts
  β(easy)  ≈ 0              # don't punish short answers on easy prompts
  β(hard)  > 0              # DO punish short answers on hard prompts
```

Then: `chosen = argmax effective_score`, `rejected = argmin effective_score`.

This is the API-only translation of LASER-D (arXiv:2505.15612) and LEASH
(arXiv:2512.21540), which use the same shape but via local gradients.

### Difficulty estimation without ground truth (API-compatible)

| Method | arXiv | Signal | Cost |
|---|---|---|---|
| Sample consistency | 2402.13904 | Disagreement across N sampled answers | N extra samples |
| Self-certainty | 2502.18581 | Distributional self-certainty from logprobs | 1 sample w/ logprobs |
| LLM-compare (Bradley-Terry) | 2512.14220 | Pairwise judge comparisons → difficulty score | judge calls |
| Verbalized self-assessment | 2605.07806 | "How hard is this for you?" self-report | 1 extra call |

**Recommended:** sample consistency (cheap, robust) — sample 4-8 answers, measure
agreement. Low agreement → hard. High agreement → easy.

---

## 4. Compute budget — small pilot

**Assumptions:** 1,000 prompts, 3 iterations, 16 samples/prompt.

### Path A: cheap model (GPT-4o-mini / Haiku-class)
| Item | Per iter | 3 iters |
|---|---|---|
| Rollouts (16k samples, 200 in / 300 out) | ~$3 | ~$9 |
| Difficulty est. (8 extra samples/prompt) | ~$1.50 | ~$4.50 |
| Judging (LLM-as-judge) | ~$1 | ~$3 |
| Fine-tuning (~2M tokens) | ~$6 | ~$18 |
| **Total** | **~$11** | **~$35** |
| Wall time | ~2-4 hr | ~6-12 hr |

### Path B: strong model (GPT-4o-class)
| Item | Per iter | 3 iters |
|---|---|---|
| Rollouts | ~$85 | ~$255 |
| Fine-tuning | ~$50 | ~$150 |
| **Total** | **~$135** | **~$405** |

**Recommendation:** start on Path A. Validate the loop end-to-end for <$50 before
scaling.

---

## 5. SSH self-modifying agent — safety architecture

Your idea: the LLM SSHes into a training box and updates its own policy. **No public
project does exactly this.** SEAL (arXiv:2506.10943) is the closest published
approximation. Here's how to do it safely:

### Tiered trust model

| Tier | Component | Access | Can the LLM touch it? |
|---|---|---|---|
| 1 | **Actor** (the LLM) | Sandbox, read-only task repo, constrained tools | — |
| 2 | **Executor** | Ephemeral container, no network to eval/train | runs actor's commands |
| 3 | **Judge/Verifier** | Immutable signed eval suite, separate network | NO |
| 4 | **Training box** | SSH from actor, but only 1 pre-approved script | YES (limited) |
| 5 | **Promotion gate** | External controller, runs hold-out eval | NO |

### The SSH tool the LLM gets

The agent's SSH key is restricted to running ONE script:
```bash
ssh train-box run_dpo_job --data /artifacts/prefs_iter_{t}.jsonl --config /etc/signed/train.cfg
```
- The agent can ONLY write the preference JSONL and trigger the job.
- It CANNOT edit the training config, the eval suite, or the judge.
- The new checkpoint is written to a read-only artifact store with a checksum.
- The promotion gate (external) decides whether to swap it in.

### Top 3 failure modes & mitigations

| # | Failure | Mitigation |
|---|---|---|
| 1 | **Reward hacking / eval tampering** — agent edits tests/grader | Immutable hidden eval; separate sandbox; outcome+process reward; red-team with RHB |
| 2 | **Prompt injection via tool output** — malicious stdout hijacks agent | Sandboxed low-priv exec; constrained tool schema; no free-form shell; InjecAgent evals |
| 3 | **Runaway self-modification** — agent edits training code/hyperparams | Signed configs; training box can't reach eval/reward; hold-out gate + auto-rollback; KL penalty vs reference model |

### Documented attacks to defend against
- **AIShellJack** (arXiv:2509.22040): 84% malicious-command execution on Cursor/Copilot via prompt injection
- **Sycophancy to Subterfuge** (arXiv:2406.10162): models learn to rewrite their own reward function
- **Reward Hacking Benchmark** (arXiv:2605.02964): DeepSeek-R1-Zero shows 13.9% exploit rate on tool-use agents

---

## 6. Self-improvement limits — when to stop

| Failure | Detection signal |
|---|---|
| Capability plateau | held-out win-rate flattens after 2-4 iters |
| Mode collapse | pass@K drops while pass@1 rises; Self-BLEU climbs |
| Length inflation | mean response length grows monotonically across iters |
| Judge-truth gap | hidden-anchor audit: judge score >> true accuracy |
| Reward hacking | GRIFT/TRACE metrics; judge-truth gap widening |

**Rule of thumb:** 2-5 iterations is the sweet spot. Beyond that, diminishing
returns + rising risk. ReST^EM plateaus at 2-3 iters; Self-Rewarding LM uses 3.

---

## 7. Key citations

### RLAIF / self-rewarding
- Self-Rewarding LM — arXiv:2401.10020 (2024)
- Constitutional AI — arXiv:2212.08073 (2022)
- DPO — arXiv:2305.18290 (2023)
- Online AI Feedback — arXiv:2402.04792 (2024)
- RAFT — arXiv:2304.06767 (2023)
- ReST — arXiv:2308.08998 (2023)
- ReST^EM — arXiv:2312.06585 (2024)
- SPIN — arXiv:2401.01335 (2024)
- SPPO — arXiv:2405.00675 (2025)
- Self-Taught Evaluators — arXiv:2408.02666 (2024)

### Overthinking / reasoning length
- "Do NOT Think That Much for 2+3=?" — arXiv:2412.21187 (2024)
- THOUGHTTERMINATOR — arXiv:2504.13367 (2025)
- Snell test-time scaling — arXiv:2408.03314 (2024)
- LASER-D — arXiv:2505.15612 (2025)
- LEASH — arXiv:2512.21540 (2025)
- Kimi k1.5 length reward — arXiv:2501.12599 (2025)
- ConciseR (L-GRPO) — arXiv:2505.21178 (2025)
- LCPO — arXiv:2508.10164 (2025)
- Dr. GRPO — arXiv:2503.20783 (2025)
- DeepSeek-R1 — arXiv:2501.12948 (2025)

### Difficulty estimation (no ground truth)
- Sample consistency — arXiv:2402.13904 (2024)
- Self-certainty — arXiv:2502.18581 (2025)
- LLM-compare — arXiv:2512.14220 (2025)

### Agentic / SSH / self-modifying
- SEAL — arXiv:2506.10943 (2025)
- ALAS — arXiv:2508.15805 (2025)
- RLEF — arXiv:2410.02089 (2024)
- SWE-RL — arXiv:2502.18449 (2025)
- LEGO-RL — arXiv:2608.17393 (2026)
- OpenHands — arXiv:2407.16741 (2024)
- Gödel Agent — arXiv:2410.04444 (2024)
- Darwin Gödel Machine — arXiv:2505.22954 (2025)

### Safety / reward hacking
- InjecAgent — arXiv:2403.02691 (2024)
- AIShellJack — arXiv:2509.22040 (2025)
- Sycophancy to Subterfuge — arXiv:2406.10162 (2024)
- Reward Hacking Benchmark — arXiv:2605.02964 (2026)
- Agent Safety Alignment via RL — arXiv:2507.08270 (2025)

### Frameworks
- TRL — github.com/huggingface/trl (local actor only)
- OpenRLHF — github.com/OpenRLHF/OpenRLHF (local actor)
- verl — github.com/volcengine/verl (vLLM rollouts, local train)
- Bespoke Curator — github.com/bespokelabsai/curator (API data gen)
- RLHFlow — github.com/RLHFlow/Online-RLHF

---

## 8. Recommended next steps

1. **Pilot on Path A** (cheap model, <$50): 1,000 prompts, 16 samples, 3 iters of
   Iterative DPO with length-difficulty-shaped preference pairs. Validate the full
   loop including the SSH fine-tune trigger.

2. **Build the difficulty estimator first** — sample-consistency over 4-8 samples.
   This is the linchpin: without a reliable easy/hard signal, the length shaping
   has no conditional to key on.

3. **Build the promotion gate before giving the agent SSH.** The gate must be
   external, immutable, and able to roll back. Do not let the LLM anywhere near
   the eval suite.

4. **Audit the judge every iteration** with a hidden anchor set scored by a
   stronger external model. Track the judge-truth gap. If it widens, stop.

5. **Track:** pass@1, pass@K, mean response length (per difficulty bucket),
   Self-BLEU, judge-truth gap, hold-out win-rate. Stop at 3-5 iters or on any
   collapse signal.

6. **Scale up** to Path B (GPT-4o-class) only after the pilot proves the loop
   is stable and the length shaping actually moves the length-vs-difficulty
   distribution in the intended direction.
