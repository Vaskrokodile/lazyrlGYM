"""
lazyRLGYM — Orchestrator
========================
The outer loop that ties everything together:

  1. Load prompts (mixed difficulty)
  2. Generate N rollouts per prompt (inference phase)
  3. Estimate difficulty per prompt (sample consistency)
  4. Judge each rollout (correctness + quality)
  5. Compute shaped rewards (length × difficulty)
  6. Build preference pairs
  7. Train with DPO (training phase)
  8. Merge LoRA → new checkpoint
  9. Evaluate on held-out set
  10. Promotion gate: keep or rollback
  11. Repeat

VRAM constraint: inference and training happen sequentially, not concurrently.
The orchestrator manages loading/unloading models between phases.
"""
import sys
import time
import json
import subprocess
import urllib.request
from pathlib import Path
from typing import Optional

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    MODEL_ID, CHECKPOINT_DIR, LOG_DIR, DATA_CACHE_DIR,
    rollout_cfg, reward_cfg, train_cfg, loop_cfg,
    THINK_TOKEN, END_THINK_TOKEN,
)
from utils.vram import unload_all, unload, get_vram_usage
from utils.logging import logger, IterationMetrics

# vLLM server config
VLLM_PORT = 8000
VLLM_API_BASE = f"http://localhost:{VLLM_PORT}/v1"
VLLM_GPU_UTIL = 0.85
WSL_DISTRO = "Ubuntu-24.04"


def _wsl_path(win_path: str) -> str:
    """Convert a Windows path to a WSL2 /mnt path."""
    p = Path(win_path)
    if p.drive:
        drive = p.drive[0].lower()
        return f"/mnt/{drive}" + str(p).replace("\\", "/")[2:]
    return str(p).replace("\\", "/")


def _is_vllm_running(port: int = VLLM_PORT) -> bool:
    """Check if vLLM server is responding."""
    try:
        urllib.request.urlopen(f"http://localhost:{port}/health", timeout=3)
        return True
    except Exception:
        return False


def _start_vllm_server(model_path: str, port: int = VLLM_PORT) -> bool:
    """Start vLLM server in WSL2. Returns True if server is ready."""
    if _is_vllm_running(port):
        print(f"  [vLLM] Server already running on port {port}")
        return True

    wsl_model = _wsl_path(model_path)
    print(f"  [vLLM] Starting server in WSL2 for model: {wsl_model}")

    # Start vLLM in WSL2 as a background process
    cmd = [
        "wsl", "-d", WSL_DISTRO, "--",
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", wsl_model,
        "--port", str(port),
        "--gpu-memory-utilization", str(VLLM_GPU_UTIL),
        "--dtype", "bfloat16",
        "--trust-remote-code",
        "--api-key", "EMPTY",
        "--max-model-len", "4096",
    ]
    print(f"  [vLLM] Command: {' '.join(cmd)}")

    # Start as a detached background process
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )

    # Wait for server to be ready (up to 3 minutes)
    print(f"  [vLLM] Waiting for server to start (up to 180s)...")
    for attempt in range(90):
        time.sleep(2)
        if _is_vllm_running(port):
            print(f"  [vLLM] Server ready after {(attempt+1)*2}s!")
            return True
        if proc.poll() is not None:
            # Process died
            print(f"  [vLLM] ERROR: Server process exited with code {proc.returncode}")
            try:
                output = proc.stdout.read(2000).decode("utf-8", errors="replace") if proc.stdout else ""
                print(f"  [vLLM] Output: {output[:1000]}")
            except Exception:
                pass
            return False
        if (attempt + 1) % 10 == 0:
            print(f"  [vLLM] Still waiting... ({(attempt+1)*2}s)")

    print(f"  [vLLM] ERROR: Server did not start within 180s")
    return False


def _stop_vllm_server(port: int = VLLM_PORT):
    """Stop vLLM server by killing the process on the port."""
    print(f"  [vLLM] Stopping server on port {port}...")
    try:
        # Kill the process in WSL2
        subprocess.run(
            ["wsl", "-d", WSL_DISTRO, "--", "bash", "-c",
             f"lsof -ti:{port} | xargs kill -9 2>/dev/null; true"],
            capture_output=True, timeout=10
        )
        time.sleep(2)
        print(f"  [vLLM] Server stopped.")
    except Exception as e:
        print(f"  [vLLM] Warning: could not stop server cleanly: {e}")


class Orchestrator:
    def __init__(
        self,
        model_path: str = MODEL_ID,
        sft_first: bool = True,
        prompt_source: str = "mixed",  # "mixed" | "fable5" | "builtin"
    ):
        self.model_path = model_path
        self.sft_first = sft_first
        self.prompt_source = prompt_source
        self.current_checkpoint: Optional[str] = None  # path to current best model
        self.iteration: int = 0
        self.heldout_prompts: list[dict] = []
        self.train_prompts: list[dict] = []
        self.vllm_running = False

    # ── Phase 0: SFT Warmup ────────────────────────────────────────────────
    def run_sft_warmup(self):
        """Optional: SFT on Fable-5 traces to teach extended reasoning."""
        print("\n" + "="*60)
        print("  PHASE 0: SFT Warmup on Fable-5 traces")
        print("="*60)

        from data.dataset import load_fable5_traces, extract_sft_pairs
        from train.sft import SFTTrainer

        try:
            traces = load_fable5_traces()
            sft_data = extract_sft_pairs(traces)
            print(f"  Loaded {len(sft_data)} SFT pairs from Fable-5")
        except Exception as e:
            print(f"  [WARN] Could not load Fable-5 traces: {e}")
            print(f"  Skipping SFT warmup.")
            return

        if not sft_data:
            print("  No SFT data available, skipping warmup.")
            return

        output_dir = CHECKPOINT_DIR / "sft_warmup"
        trainer = SFTTrainer(self.model_path, train_cfg)
        try:
            result = trainer.train(sft_data[:200], output_dir)  # limit for pilot
            print(f"  SFT complete: loss={result['loss']:.4f}")
            self.current_checkpoint = str(output_dir)
        except Exception as e:
            print(f"  [ERROR] SFT failed: {e}")
            self.current_checkpoint = self.model_path
        finally:
            trainer.unload()
            unload_all()

    # ── Phase 1: Rollout Generation ────────────────────────────────────────
    def run_rollouts(self, prompts: list[dict]) -> list[dict]:
        """Generate N completions per prompt. Returns flat list of rollouts."""
        print("\n  [Phase 1] Generating rollouts (transformers)...")

        from inference.rollout import RolloutGenerator

        checkpoint = self.current_checkpoint or self.model_path
        gen = RolloutGenerator(checkpoint)

        all_rollouts = []
        try:
            # Use batched generation for speed: process multiple prompts per
            # model.generate() call instead of one-at-a-time.
            prompt_texts = [p["text"] for p in prompts]
            batch_results = gen.generate_batch(
                prompts=prompt_texts,
                n_samples=rollout_cfg.n_samples_per_prompt,
                max_new_tokens=rollout_cfg.max_new_tokens,
                temperature=rollout_cfg.temperature,
                top_p=rollout_cfg.top_p,
                batch_size=rollout_cfg.batch_size,
            )

            for i, (prompt_dict, samples) in enumerate(zip(prompts, batch_results)):
                prompt = prompt_dict["text"]
                prompt_id = prompt_dict["id"]

                for j, s in enumerate(samples):
                    # Build a formatted response with think markers for DPO.
                    if s["thinking_text"]:
                        response_formatted = (
                            f"{THINK_TOKEN}\n{s['thinking_text']}\n"
                            f"{END_THINK_TOKEN}\n{s['answer_text']}"
                        )
                    else:
                        response_formatted = s["answer_text"]

                    all_rollouts.append({
                        "prompt_id": prompt_id,
                        "prompt": prompt,
                        "response": response_formatted,
                        "response_raw": s["text"],
                        "thinking_text": s["thinking_text"],
                        "answer_text": s["answer_text"],
                        "thinking_tokens": s["thinking_tokens"],
                        "answer_tokens": s["answer_tokens"],
                        "total_tokens": s["total_tokens"],
                        "sample_idx": j,
                        "difficulty_hint": prompt_dict.get("difficulty_hint", "medium"),
                    })

                if (i + 1) % 10 == 0 or (i + 1) == len(prompts):
                    print(f"    {i+1}/{len(prompts)} prompts done | VRAM: {get_vram_usage()['allocated']:.0f}MB")
        finally:
            gen.unload()
            unload_all()

        print(f"  Generated {len(all_rollouts)} rollouts from {len(prompts)} prompts")
        return all_rollouts

    # ── Phase 2: Difficulty Estimation ─────────────────────────────────────
    def estimate_difficulties(self, rollouts: list[dict]) -> dict[str, tuple[str, float]]:
        """Estimate difficulty per prompt using sample consistency."""
        print("\n  [Phase 2] Estimating prompt difficulty...")

        from reward.difficulty import estimate_difficulty

        # Group rollouts by prompt_id
        by_prompt: dict[str, list[dict]] = {}
        for r in rollouts:
            by_prompt.setdefault(r["prompt_id"], []).append(r)

        difficulties = {}
        for pid, rolls in by_prompt.items():
            responses = [r["answer_text"] for r in rolls]
            diff, conf = estimate_difficulty(responses)
            difficulties[pid] = (diff, conf)

        # Print distribution
        counts = {"easy": 0, "medium": 0, "hard": 0}
        for d, _ in difficulties.values():
            counts[d] = counts.get(d, 0) + 1
        print(f"    Difficulty distribution: {counts}")

        return difficulties

    # ── Phase 3: Judging ───────────────────────────────────────────────────
    def _build_expected_answers(self) -> dict[str, str]:
        """Build a {prompt_text: expected_answer} dict from all loaded prompts."""
        expected = {}
        for p in self.train_prompts + self.heldout_prompts:
            ans = p.get("expected_answer")
            if ans:
                expected[p["text"]] = ans
        return expected

    def _make_judge(self):
        """Create a judge instance, passing expected_answers for rule-based mode."""
        from reward.judge import get_judge

        mode = loop_cfg.judge_mode
        if mode == "rule":
            return get_judge(
                mode=mode,
                expected_answers=self._build_expected_answers(),
            )
        return get_judge(
            mode=mode,
            model_path=self.current_checkpoint or self.model_path,
            provider=loop_cfg.judge_api_provider,
            api_key_env=loop_cfg.judge_api_key_env,
        )

    def judge_rollouts(self, rollouts: list[dict], difficulties: dict) -> list[dict]:
        """Score each rollout with the judge."""
        print("\n  [Phase 3] Judging rollouts...")

        judge = self._make_judge()

        scored = []
        try:
            for i, r in enumerate(rollouts):
                diff = difficulties.get(r["prompt_id"], ("medium", 0.5))[0]
                scores = judge.score(
                    prompt=r["prompt"],
                    response=r["answer_text"],
                    thinking=r["thinking_text"],
                )
                r["correctness"] = scores["correctness"]
                r["quality"] = scores["quality"]
                r["difficulty"] = diff
                scored.append(r)

                if (i + 1) % 20 == 0:
                    print(f"    Judged {i+1}/{len(rollouts)}")
        finally:
            if hasattr(judge, "unload"):
                judge.unload()
            unload_all()

        print(f"  Judged {len(scored)} rollouts")
        return scored

    # ── Phase 4: Reward Shaping ────────────────────────────────────────────
    def compute_rewards(self, scored_rollouts: list[dict]) -> list[dict]:
        """Apply length × difficulty reward shaping."""
        print("\n  [Phase 4] Computing shaped rewards...")

        from reward.shaping import compute_shaped_reward

        for r in scored_rollouts:
            r["shaped_reward"] = compute_shaped_reward(
                correctness=r["correctness"],
                quality=r["quality"],
                thinking_tokens=r["thinking_tokens"],
                difficulty=r["difficulty"],
                cfg=reward_cfg,
            )

        # Print reward stats
        rewards = [r["shaped_reward"] for r in scored_rollouts]
        print(f"    Mean reward: {sum(rewards)/len(rewards):.3f}")
        print(f"    Min/Max: {min(rewards):.3f} / {max(rewards):.3f}")

        return scored_rollouts

    # ── Phase 5: Build Preference Pairs ────────────────────────────────────
    def build_preferences(self, scored_rollouts: list[dict]) -> list[dict]:
        """Build DPO preference pairs from scored rollouts."""
        print("\n  [Phase 5] Building preference pairs...")

        from data.preferences import build_preference_pairs, save_preference_dataset

        rewards = [r["shaped_reward"] for r in scored_rollouts]
        pairs = build_preference_pairs(scored_rollouts, rewards)

        # Save for inspection
        path = DATA_CACHE_DIR / f"prefs_iter_{self.iteration}.jsonl"
        save_preference_dataset(pairs, path)
        print(f"    Built {len(pairs)} preference pairs -> {path}")

        return pairs

    # ── Phase 6: DPO Training ──────────────────────────────────────────────
    def run_dpo_training(self, pairs: list[dict]) -> dict:
        """Train with DPO on the preference pairs."""
        print("\n  [Phase 6] DPO training...")

        from train.dpo import DPOTrainer
        from train.merge import merge_and_save

        checkpoint = self.current_checkpoint or self.model_path
        output_dir = CHECKPOINT_DIR / f"dpo_iter_{self.iteration}"
        merged_dir = CHECKPOINT_DIR / f"merged_iter_{self.iteration}"

        trainer = DPOTrainer(checkpoint, train_cfg)
        try:
            result = trainer.train(pairs, output_dir)
            print(f"    DPO loss: {result['loss']:.4f}  reward_acc: {result.get('reward_accuracy', 0):.3f}")

            # Merge LoRA into a standalone model for inference
            merge_and_save(checkpoint, str(output_dir), str(merged_dir))
            print(f"    Merged model -> {merged_dir}")

            return {
                "loss": result["loss"],
                "reward_accuracy": result.get("reward_accuracy", 0),
                "adapter_dir": str(output_dir),
                "merged_dir": str(merged_dir),
            }
        except Exception as e:
            print(f"    [ERROR] DPO training failed: {e}")
            raise
        finally:
            trainer.unload()
            unload_all()

    # ── Phase 7: Held-out Evaluation ───────────────────────────────────────
    def evaluate_heldout(self, candidate_dir: str) -> dict:
        """Evaluate the new checkpoint on held-out prompts."""
        print("\n  [Phase 7] Held-out evaluation...")

        from inference.eval import Evaluator

        evaluator = Evaluator(candidate_dir)
        judge = self._make_judge()

        # Wrapper: judge.score() returns a dict; eval expects a float score.
        def _judge_score_fn(prompt: str, response: str) -> float:
            result = judge.score(prompt, response)
            return result["correctness"] * 0.7 + result["quality"] * 0.3

        try:
            # Use 1024 max_new_tokens for eval (matches rollout config, faster than 2048)
            results = evaluator.evaluate_heldout(
                self.heldout_prompts, _judge_score_fn,
                max_new_tokens=1024,
            )

            # Only run pass@k on first and last iteration for speed
            is_boundary = (self.iteration == 0 or self.iteration >= loop_cfg.max_iterations - 1)
            if is_boundary or not loop_cfg.pass_at_k_only_boundary:
                pass_at_k = evaluator.compute_pass_at_k(
                    self.heldout_prompts[:20],
                    k=loop_cfg.pass_at_k_k,
                    judge_fn=_judge_score_fn,
                    max_new_tokens=1024,
                )
            else:
                # Skip pass@k on non-boundary iterations — reuse previous value
                pass_at_k = -1.0  # sentinel: "not computed this iteration"
            results["pass_at_k"] = pass_at_k
            print(f"    Winrate: {results['winrate']:.3f}  Score: {results['mean_score']:.3f}")
            if pass_at_k >= 0:
                print(f"    Pass@K: {pass_at_k:.3f}  Mean length: {results['mean_length']:.0f}")
            else:
                print(f"    Pass@K: skipped (non-boundary)  Mean length: {results['mean_length']:.0f}")
            return results
        finally:
            evaluator.unload()
            if hasattr(judge, "unload"):
                judge.unload()
            unload_all()

    # ── Phase 8: Promotion Gate ────────────────────────────────────────────
    def promotion_gate(
        self,
        eval_results: dict,
        prev_eval: Optional[dict],
        length_stats: dict,
    ) -> tuple[bool, str]:
        """Decide whether to promote the new checkpoint or rollback."""
        print("\n  [Phase 8] Promotion gate...")

        pass_at_k = eval_results.get("pass_at_k", 1.0)

        if prev_eval is None:
            # First iteration — always promote as baseline (even if pass@K is low)
            return True, "first iteration (baseline)"

        # Check improvement (use score, not just winrate, for finer signal)
        delta = eval_results["mean_score"] - prev_eval["mean_score"]
        if delta < loop_cfg.min_heldout_improvement:
            # Score didn't improve — also check if length exploded
            if "mean_length" in eval_results and "mean_length" in prev_eval:
                ratio = eval_results["mean_length"] / max(prev_eval["mean_length"], 1)
                if ratio > loop_cfg.max_length_growth_ratio:
                    return False, f"no improvement (delta={delta:.3f}) and length explosion: ratio={ratio:.2f}"
            return False, f"no improvement: delta={delta:.3f}"

        # Score improved — only block on severe length explosion
        if "mean_length" in eval_results and "mean_length" in prev_eval:
            ratio = eval_results["mean_length"] / max(prev_eval["mean_length"], 1)
            if ratio > loop_cfg.max_length_growth_ratio:
                return False, f"length explosion: ratio={ratio:.2f}"

        # Check diversity (only if pass@k was computed this iteration)
        if pass_at_k >= 0 and pass_at_k < loop_cfg.min_pass_at_k:
            return False, f"diversity collapse: pass@K={pass_at_k:.3f}"

        return True, f"improved by {delta:.3f}"

    # ── Prompt Loading ─────────────────────────────────────────────────────
    def load_prompts(self):
        """Load training and held-out prompts."""
        from data.prompts import get_prompts, load_fable5_prompts

        all_prompts = []

        if self.prompt_source in ("mixed", "builtin"):
            builtin = get_prompts(n=loop_cfg.prompts_per_iteration)
            all_prompts.extend(builtin)

        if self.prompt_source in ("mixed", "fable5"):
            try:
                fable5 = load_fable5_prompts()
                all_prompts.extend(fable5[:50])  # add some hard prompts
            except Exception as e:
                print(f"  [WARN] Could not load Fable-5 prompts: {e}")

        # Split: 80% train, 20% held-out
        split = int(len(all_prompts) * 0.8)
        self.train_prompts = all_prompts[:split]
        self.heldout_prompts = all_prompts[split:]
        print(f"  Loaded {len(all_prompts)} prompts ({len(self.train_prompts)} train, {len(self.heldout_prompts)} held-out)")

    # ── Main Loop ──────────────────────────────────────────────────────────
    def run(self):
        """Run the full training loop."""
        print("\n" + "#"*60)
        print("#  lazyRLGYM — Starting RL Training Loop")
        print(f"#  Model: {self.model_path}")
        print(f"#  Max iterations: {loop_cfg.max_iterations}")
        print(f"#  Judge mode: {loop_cfg.judge_mode}")
        print("#"*60 + "\n")

        # Load prompts
        self.load_prompts()

        # Optional SFT warmup
        if self.sft_first:
            self.run_sft_warmup()

        prev_eval = None

        for self.iteration in range(loop_cfg.max_iterations):
            print(f"\n{'#'*60}")
            print(f"#  ITERATION {self.iteration}")
            print(f"{'#'*60}")

            metrics = IterationMetrics(iteration=self.iteration)
            iter_start = time.time()

            try:
                # 1. Rollouts (loads model for inference)
                rollouts = self.run_rollouts(self.train_prompts)
                metrics.n_prompts = len(self.train_prompts)
                metrics.n_samples = len(rollouts)
                metrics.total_tokens_generated = sum(r["total_tokens"] for r in rollouts)

                # 2. Difficulty
                difficulties = self.estimate_difficulties(rollouts)
                for d, _ in difficulties.values():
                    if d == "easy": metrics.n_easy += 1
                    elif d == "hard": metrics.n_hard += 1
                    else: metrics.n_medium += 1

                # 3. Judge (loads model again for judging)
                scored = self.judge_rollouts(rollouts, difficulties)
                metrics.mean_correctness = sum(r["correctness"] for r in scored) / len(scored)
                metrics.mean_quality = sum(r["quality"] for r in scored) / len(scored)

                # 4. Rewards
                scored = self.compute_rewards(scored)
                metrics.mean_reward = sum(r["shaped_reward"] for r in scored) / len(scored)

                # Length stats per difficulty
                from reward.shaping import analyze_length_distribution
                length_stats = analyze_length_distribution(scored, [r["difficulty"] for r in scored])
                metrics.mean_length_easy = length_stats.get("easy", {}).get("mean", 0)
                metrics.mean_length_medium = length_stats.get("medium", {}).get("mean", 0)
                metrics.mean_length_hard = length_stats.get("hard", {}).get("mean", 0)

                # 5. Preference pairs
                pairs = self.build_preferences(scored)

                # 6. DPO training (VRAM is free — inference model was unloaded)
                train_result = self.run_dpo_training(pairs)
                metrics.train_loss = train_result["loss"]

                # 7. Evaluate (loads merged model for inference)
                eval_results = self.evaluate_heldout(train_result["merged_dir"])
                metrics.heldout_winrate = eval_results["winrate"]
                metrics.heldout_score = eval_results["mean_score"]
                metrics.pass_at_1 = eval_results.get("pass_at_1", 0)
                metrics.pass_at_k = eval_results.get("pass_at_k", 0)

                # 8. Promotion gate
                promoted, reason = self.promotion_gate(eval_results, prev_eval, length_stats)
                metrics.promoted = promoted
                metrics.rollback_reason = reason

                if promoted:
                    self.current_checkpoint = train_result["merged_dir"]
                    prev_eval = eval_results
                    print(f"\n  [PROMOTED] CHECKPOINT PROMOTED: {reason}")
                else:
                    print(f"\n  [ROLLBACK] ROLLING BACK: {reason}")

                # Log
                logger.log_iteration(metrics)
                elapsed = time.time() - iter_start
                print(f"\n  Iteration {self.iteration} completed in {elapsed:.0f}s")

            except Exception as e:
                print(f"\n  [ERROR] Iteration {self.iteration} failed: {e}")
                import traceback
                traceback.print_exc()
                metrics.rollback_reason = f"error: {str(e)[:100]}"
                logger.log_iteration(metrics)
                unload_all()
                continue

        print("\n" + "#"*60)
        print("#  lazyRLGYM — Training Complete")
        print(f"#  Final checkpoint: {self.current_checkpoint}")
        print("#"*60)

        return self.current_checkpoint


# ── Entry Point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    # Default to local model cache if it exists
    local_model = r"E:\lazyRLGYM\data\model_cache"
    default_model = local_model if Path(local_model).exists() else MODEL_ID

    parser = argparse.ArgumentParser(description="lazyRLGYM RL Training Loop")
    parser.add_argument("--model", default=default_model, help="Model path or HF ID")
    parser.add_argument("--no-sft", action="store_true", help="Skip SFT warmup")
    parser.add_argument("--prompts", default="mixed", choices=["mixed", "fable5", "builtin"])
    parser.add_argument("--judge", default="local", choices=["local", "api", "rule"])
    parser.add_argument("--iterations", type=int, default=None, help="Override max iterations")
    args = parser.parse_args()

    if args.iterations:
        loop_cfg.max_iterations = args.iterations
    loop_cfg.judge_mode = args.judge

    orch = Orchestrator(
        model_path=args.model,
        sft_first=not args.no_sft,
        prompt_source=args.prompts,
    )
    orch.run()
