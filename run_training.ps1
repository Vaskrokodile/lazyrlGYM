# lazyRLGYM — Training Runner with Memory Safety
# Sets environment variables for VRAM/RAM safety and launches the orchestrator.
$ErrorActionPreference = "Continue"

# ── Memory-friendly environment ─────────────────────────────────────────────
$env:TOKENIZERS_PARALLELISM = "false"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"

# ── Configuration ───────────────────────────────────────────────────────────
$LogFile = "E:\lazyRLGYM\logs\training_run.log"
$Timestamp = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"

Write-Output "[$Timestamp] Starting lazyRLGYM training run..."
Write-Output "[$Timestamp] Log file: $LogFile"

# ── Launch orchestrator ─────────────────────────────────────────────────────
Set-Location E:\lazyRLGYM

# Run with: no SFT warmup (faster), rule-based judge (no extra model), builtin prompts only
python -u orchestrator.py --no-sft --judge rule --prompts builtin --iterations 3 *>&1 | Out-File -FilePath $LogFile -Encoding utf8

$EndTime = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
Write-Output "[$EndTime] Training run finished. Check $LogFile for details."
