"""Quick test to verify the rollout thinking token extraction fix."""
import sys
import os
from pathlib import Path

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONIOENCODING"] = "utf-8"
sys.path.insert(0, str(Path(__file__).parent))

from inference.rollout import RolloutGenerator

MODEL_PATH = r"E:\lazyRLGYM\data\model_cache"
gen = RolloutGenerator(MODEL_PATH)

print("\nGenerating 2 samples for 'What is 2+3?'...")
results = gen.generate(
    prompt="What is 2 + 3?",
    n_samples=2,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.95,
)

for i, s in enumerate(results):
    print(f"\n--- Sample {i} ---")
    print(f"  thinking_tokens: {s['thinking_tokens']}")
    print(f"  answer_tokens: {s['answer_tokens']}")
    print(f"  total_tokens: {s['total_tokens']}")
    print(f"  thinking_text (first 200 chars): {s['thinking_text'][:200]}")
    print(f"  answer_text (first 200 chars): {s['answer_text'][:200]}")

gen.unload()
print("\nTest complete!")
