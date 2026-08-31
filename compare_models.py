"""Compare base model vs trained checkpoint on the same prompts."""
import os
import sys
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(Path(__file__).parent))

from inference.rollout import RolloutGenerator

BASE_MODEL = r"E:\lazyRLGYM\data\model_cache"
TRAINED_MODEL = r"E:\lazyRLGYM\checkpoints\merged_iter_0"

PROMPTS = [
    ("easy", "What is 2 + 3?"),
    ("easy", "What is 7 multiplied by 6?"),
    ("medium", "A train travels 60 km in 45 minutes. What is its average speed in km/h?"),
    ("hard", "Design the system architecture for a URL shortener service that handles 100 million URLs and 200 reads per second."),
]

def run_model(model_path, label, prompts):
    print(f"\n{'='*70}")
    print(f"  {label}: {model_path}")
    print(f"{'='*70}")
    gen = RolloutGenerator(model_path)
    results = {}
    for diff, prompt in prompts:
        samples = gen.generate(
            prompt=prompt, n_samples=1,
            max_new_tokens=1024, temperature=0.7, top_p=0.95,
        )
        s = samples[0]
        results[(diff, prompt)] = s
        print(f"\n--- [{diff.upper()}] {prompt[:60]}... ---")
        print(f"  thinking_tokens: {s['thinking_tokens']}  answer_tokens: {s['answer_tokens']}  total: {s['total_tokens']}")
        print(f"  THINKING (first 300 chars): {s['thinking_text'][:300]}")
        print(f"  ANSWER (first 200 chars): {s['answer_text'][:200]}")
    gen.unload()
    return results

# Run both models
base_results = run_model(BASE_MODEL, "BASE MODEL (untrained)", PROMPTS)
trained_results = run_model(TRAINED_MODEL, "TRAINED MODEL (iter 0)", PROMPTS)

# Summary comparison
print(f"\n\n{'='*70}")
print("  SUMMARY: Thinking token comparison")
print(f"{'='*70}")
print(f"{'Difficulty':<10} {'Base tokens':>15} {'Trained tokens':>15} {'Delta':>10}")
print("-" * 55)
for diff, prompt in PROMPTS:
    b = base_results[(diff, prompt)]
    t = trained_results[(diff, prompt)]
    print(f"{diff:<10} {b['thinking_tokens']:>15} {t['thinking_tokens']:>15} {t['thinking_tokens'] - b['thinking_tokens']:>+10}")
print()
