"""Extract and display iteration summaries from the training log."""
import re
import sys
from pathlib import Path

log_path = Path(r"E:\lazyRLGYM\logs\training_run.log")
if not log_path.exists():
    print("No log file found yet.")
    sys.exit(0)

text = log_path.read_text(encoding="utf-8")

# Find all iteration summary blocks
# Pattern: "Iteration N — timestamp" followed by metrics until "===="
pattern = r"(Iteration \d+).*?\n=+\n(.*?)\n=+"
matches = re.findall(pattern, text, re.DOTALL)

if not matches:
    print("No completed iterations yet. Training may still be in early phases.")
    # Show what phase we're in
    lines = text.strip().split("\n")
    recent = [l for l in lines[-20:] if l.strip() and not l.startswith("  0%") and not l.startswith("Loading")]
    print("\n--- Last 15 meaningful log lines ---")
    for l in recent[-15:]:
        print(l.strip())
    sys.exit(0)

print(f"Found {len(matches)} completed iteration(s):\n")

for iter_name, metrics_block in matches:
    print(f"{'='*60}")
    print(f"  {iter_name.strip()}")
    print(f"{'='*60}")
    for line in metrics_block.strip().split("\n"):
        line = line.strip()
        if line:
            print(f"  {line}")
    print()

# Also show current phase if training is still running
lines = text.strip().split("\n")
# Find last "ITERATION N" header
iter_headers = [(i, l) for i, l in enumerate(lines) if "ITERATION" in l and "#" in l]
if iter_headers:
    last_iter_line_idx = iter_headers[-1][0]
    # Check if there's a completed summary after this
    after_last = "\n".join(lines[last_iter_line_idx:])
    if "Decision:" not in after_last:
        # This iteration is still running — show current phase
        print(f"\n{'='*60}")
        print("  CURRENT (in progress):")
        print(f"{'='*60}")
        recent = [l for l in lines[last_iter_line_idx:]
                  if l.strip()
                  and not l.startswith("  0%")
                  and not "Loading checkpoint" in l
                  and not "examples/s" in l
                  and not "it/s" in l]
        for l in recent[-10:]:
            print(f"  {l.strip()}")
