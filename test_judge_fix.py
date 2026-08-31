"""Quick test to verify the judge fix works."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from data.prompts import get_prompts
from reward.judge import RuleBasedJudge

# Load prompts
prompts = get_prompts(n=12)
print(f"Loaded {len(prompts)} prompts")

# Build expected answers
expected = {}
for p in prompts:
    ans = p.get("expected_answer")
    if ans:
        expected[p["text"]] = ans
        print(f"  [{p['id']}] expected: {ans}")

print(f"\nTotal expected answers: {len(expected)}")

# Test judge with expected answers
judge = RuleBasedJudge(expected_answers=expected)

# Test: correct answer for 2+3
result = judge.score(
    prompt="What is 2 + 3?",
    response="The answer is 5.",
    thinking="I need to add 2 and 3.",
)
print(f'\nJudge test (correct 2+3=5): correctness={result["correctness"]:.2f} quality={result["quality"]:.2f}')

# Test: wrong answer for 2+3
result = judge.score(
    prompt="What is 2 + 3?",
    response="The answer is 7.",
    thinking="I need to add 2 and 3.",
)
print(f'Judge test (wrong 2+3=7):   correctness={result["correctness"]:.2f} quality={result["quality"]:.2f}')

# Test: correct answer for 7*6
result = judge.score(
    prompt="What is 7 multiplied by 6?",
    response="The answer is 42.",
    thinking="7 times 6.",
)
print(f"Judge test (correct 7*6=42): correctness={result['correctness']:.2f} quality={result['quality']:.2f}")

# Test: no expected answer (coding prompt)
result = judge.score(
    prompt="Write a Python function to reverse a string.",
    response="def reverse(s): return s[::-1]",
    thinking="I need to reverse a string.",
)
print(f"Judge test (coding, no expected): correctness={result['correctness']:.2f} quality={result['quality']:.2f}")

print("\nAll judge tests passed!")
