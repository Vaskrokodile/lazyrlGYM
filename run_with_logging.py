"""Wrapper to run the orchestrator with real-time file logging."""
import sys
import os
from pathlib import Path

# Set environment before importing anything
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"

# Redirect stdout/stderr to a log file with line buffering
log_path = Path(__file__).parent / "logs" / "training_run.log"
log_path.parent.mkdir(parents=True, exist_ok=True)

log_file = open(log_path, "w", encoding="utf-8", buffering=1)  # line-buffered
sys.stdout = log_file
sys.stderr = log_file

# Now run the orchestrator
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)


def main():
    # Import and run
    from orchestrator import Orchestrator
    from config import loop_cfg

    local_model = r"E:\lazyRLGYM\data\model_cache"
    default_model = local_model if Path(local_model).exists() else "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    loop_cfg.max_iterations = 5
    loop_cfg.judge_mode = "rule"

    orch = Orchestrator(
        model_path=default_model,
        sft_first=False,
        prompt_source="builtin",
    )
    orch.run()

    log_file.close()


if __name__ == "__main__":
    main()
