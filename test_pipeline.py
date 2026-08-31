"""End-to-end smoke test for lazyRLGYM pipeline."""
import sys
import os
from pathlib import Path

# Setup path
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

print("=" * 60)
print("  lazyRLGYM — End-to-End Smoke Test")
print("=" * 60)

MODEL_PATH = r"E:\lazyRLGYM\data\model_cache"

# ── Test 1: Model loads and generates ──────────────────────────────────────
print("\n[Test 1] Loading model and generating a response...")
from inference.rollout import RolloutGenerator

gen = RolloutGenerator(MODEL_PATH)
print("  Model loaded successfully.")

results = gen.generate(
    prompt="What is 2+3?",
    n_samples=2,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
)
print(f"  Generated {len(results)} samples")
for i, s in enumerate(results):
    print(f"  Sample {i}: thinking={s['thinking_tokens']}tok answer={s['answer_tokens']}tok")
    print(f"    Thinking: {s['thinking_text'][:150]}...")
    print(f"    Answer: {s['answer_text'][:150]}")

gen.unload()
print("  [PASS] Model loads and generates.\n")

# ── Test 2: Difficulty estimation ──────────────────────────────────────────
print("[Test 2] Difficulty estimation...")
from reward.difficulty import estimate_difficulty

# Easy: all answers agree
easy_responses = ["The answer is 5.", "5", "The answer is 5."]
diff, conf = estimate_difficulty(easy_responses)
print(f"  Easy test: difficulty={diff}, confidence={conf:.2f}")
assert diff == "easy", f"Expected easy, got {diff}"

# Hard: answers disagree
hard_responses = ["The answer is 42", "The answer is 17", "The answer is 99"]
diff, conf = estimate_difficulty(hard_responses)
print(f"  Hard test: difficulty={diff}, confidence={conf:.2f}")
assert diff == "hard", f"Expected hard, got {diff}"

print("  [PASS] Difficulty estimation works.\n")

# ── Test 3: Reward shaping ─────────────────────────────────────────────────
print("[Test 3] Reward shaping...")
from reward.shaping import compute_shaped_reward
from config import reward_cfg

# Easy prompt, short correct answer → should get positive reward
r1 = compute_shaped_reward(correctness=1.0, quality=0.9, thinking_tokens=50, difficulty="easy", cfg=reward_cfg)
print(f"  Easy+short+correct: reward={r1:.3f}")

# Easy prompt, very long thinking → should be penalized
r2 = compute_shaped_reward(correctness=1.0, quality=0.9, thinking_tokens=2000, difficulty="easy", cfg=reward_cfg)
print(f"  Easy+long+correct:  reward={r2:.3f}")
assert r2 < r1, "Overthinking on easy should be penalized!"

# Hard prompt, short thinking → should be penalized for underthinking
r3 = compute_shaped_reward(correctness=0.5, quality=0.3, thinking_tokens=50, difficulty="hard", cfg=reward_cfg)
print(f"  Hard+short:         reward={r3:.3f}")

# Hard prompt, optimal thinking → should get better reward
r4 = compute_shaped_reward(correctness=1.0, quality=0.9, thinking_tokens=2000, difficulty="hard", cfg=reward_cfg)
print(f"  Hard+optimal+correct: reward={r4:.3f}")
assert r4 > r3, "Optimal thinking on hard should beat underthinking!"

print("  [PASS] Reward shaping works correctly.\n")

# ── Test 4: Preference pair construction ───────────────────────────────────
print("[Test 4] Preference pair construction...")
from data.preferences import build_preference_pairs

rollouts = [
    {"prompt_id": "p1", "prompt": "What is 2+3?", "response": "5", "thinking_tokens": 10, "answer_tokens": 1, "difficulty": "easy"},
    {"prompt_id": "p1", "prompt": "What is 2+3?", "response": "Let me think... 5", "thinking_tokens": 500, "answer_tokens": 5, "difficulty": "easy"},
    {"prompt_id": "p1", "prompt": "What is 2+3?", "response": "7", "thinking_tokens": 100, "answer_tokens": 1, "difficulty": "easy"},
]
scores = [1.5, 0.3, 0.1]  # short correct > long correct > wrong
pairs = build_preference_pairs(rollouts, scores)
print(f"  Built {len(pairs)} preference pairs")
assert len(pairs) == 1, f"Expected 1 pair, got {len(pairs)}"
print(f"  Chosen score: {pairs[0]['chosen_score']:.2f}")
print(f"  Rejected score: {pairs[0]['rejected_score']:.2f}")
assert pairs[0]["chosen"] == "5", "Chosen should be the short correct answer"
print("  [PASS] Preference pairs work.\n")

# ── Test 5: Prompt bank ────────────────────────────────────────────────────
print("[Test 5] Prompt bank...")
from data.prompts import get_prompts

prompts = get_prompts(n=10)
print(f"  Got {len(prompts)} prompts")
diffs = [p["difficulty_hint"] for p in prompts]
print(f"  Difficulties: {diffs}")
assert len(prompts) == 10
print("  [PASS] Prompt bank works.\n")

# ── Test 6: Rule-based judge ───────────────────────────────────────────────
print("[Test 6] Rule-based judge...")
from reward.judge import RuleBasedJudge

judge = RuleBasedJudge()
# Test math judging
result = judge.score(
    prompt="What is 2+3?",
    response="The answer is 5.",
    thinking="I need to add 2 and 3.",
)
print(f"  Math judge: correctness={result['correctness']:.2f} quality={result['quality']:.2f}")
print("  [PASS] Rule-based judge works.\n")

# ── Summary ────────────────────────────────────────────────────────────────
print("=" * 60)
print("  ALL TESTS PASSED")
print("  Pipeline is ready to run!")
print("=" * 60)
print("\n  To start training:")
print("    cd E:\\lazyRLGYM")
print("    python orchestrator.py --no-sft --judge rule --iterations 1")
