#!/usr/bin/env python3
"""
lazyRLGYM — vLLM Server Management
===================================
Start / stop / monitor a vLLM OpenAI-compatible inference server.

This script is designed to run **inside WSL2** (Ubuntu-24.04) where the
GPU and the ``vllm`` package are available.  The model weights live on
the Windows drive and are accessed via ``/mnt/e/...``.

Typical invocation from Windows::

    wsl -d Ubuntu-24.04 -- python3 /mnt/e/lazyRLGYM/inference/vllm_server.py \
        --model /mnt/e/lazyRLGYM/data/model_cache --port 8000

It can also be imported as a module so the Windows-side orchestrator can
launch the server programmatically via ``subprocess``.

Functions
---------
  * ``start_vllm_server(model_path, port, gpu_memory_utilization)``
        -> subprocess.Popen
  * ``stop_vllm_server(process)``
  * ``wait_for_server(url, timeout)``
  * ``is_server_running(url)``
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
import urllib.request
import urllib.error


# ── Health checks ───────────────────────────────────────────────────────────

def is_server_running(url: str = "http://localhost:8000") -> bool:
    """Return True if the vLLM server responds at the given base URL.

    Polls the ``/health`` endpoint (vLLM's OpenAI server exposes this).
    """
    health_url = url.rstrip("/") + "/health"
    try:
        req = urllib.request.Request(health_url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return False
    except Exception:
        return False


def wait_for_server(url: str = "http://localhost:8000", timeout: int = 120) -> bool:
    """Poll the health endpoint until the server is ready or timeout.

    Returns True once healthy, False if the timeout is reached.
    """
    health_url = url.rstrip("/") + "/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(health_url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            pass
        except Exception:
            pass
        time.sleep(2)
    return False


# ── Start / stop ────────────────────────────────────────────────────────────

def start_vllm_server(
    model_path: str,
    port: int = 8000,
    gpu_memory_utilization: float = 0.85,
    dtype: str = "bfloat16",
    max_model_len: int | None = None,
    extra_args: list[str] | None = None,
) -> subprocess.Popen:
    """Start a vLLM OpenAI-compatible server as a subprocess.

    Parameters
    ----------
    model_path : str
        Path (as seen from WSL2) to the model weights, e.g.
        ``/mnt/e/lazyRLGYM/data/model_cache``.
    port : int
        TCP port to serve on.
    gpu_memory_utilization : float
        Fraction of GPU memory vLLM may use (0.0–1.0).
    dtype : str
        Model dtype (bfloat16, float16, …).
    max_model_len : int | None
        Optional maximum context length.  If None, vLLM uses the
        model's configured maximum.
    extra_args : list[str] | None
        Additional raw CLI flags appended to the vLLM command.

    Returns
    -------
    subprocess.Popen
        The handle for the running vLLM process.  The function blocks
        until the server reports healthy (or raises on timeout).
    """
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
        "--dtype", dtype,
        "--trust-remote-code",
        "--api-key", "EMPTY",
    ]
    if max_model_len is not None:
        cmd += ["--max-model-len", str(max_model_len)]
    if extra_args:
        cmd += extra_args

    print(f"[vLLM] Starting server: {' '.join(cmd)}")
    # Inherit stdout/stderr so logs are visible.
    proc = subprocess.Popen(
        cmd,
        stdout=sys.stdout if sys.stdout else subprocess.PIPE,
        stderr=sys.stderr if sys.stderr else subprocess.STDOUT,
    )

    base_url = f"http://localhost:{port}"
    print(f"[vLLM] Waiting for server at {base_url} ...")
    if not wait_for_server(base_url, timeout=180):
        # Best-effort cleanup before raising.
        stop_vllm_server(proc)
        raise RuntimeError(
            f"[vLLM] Server did not become healthy within 180s at {base_url}."
        )
    print(f"[vLLM] Server is ready at {base_url}.")
    return proc


def stop_vllm_server(process: subprocess.Popen) -> None:
    """Terminate a vLLM server process gracefully, then forcefully."""
    if process is None:
        return
    if process.poll() is not None:
        print(f"[vLLM] Process {process.pid} already exited.")
        return

    print(f"[vLLM] Stopping server (pid={process.pid}) ...")
    try:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            print("[vLLM] Graceful terminate timed out; killing.")
            process.kill()
            process.wait(timeout=10)
    except Exception as e:
        print(f"[vLLM] Error stopping server: {e}")
        try:
            process.kill()
        except Exception:
            pass
    print("[vLLM] Server stopped.")


# ── CLI entry point ─────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Start a vLLM OpenAI-compatible server for lazyRLGYM."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Path to the model weights (WSL2 path, e.g. /mnt/e/lazyRLGYM/data/model_cache).",
    )
    parser.add_argument("--port", type=int, default=8000, help="Server port.")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of GPU memory vLLM may use (default 0.85).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Model dtype (default bfloat16).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Maximum context length (default: model's own max).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Seconds to wait for the server to become healthy.",
    )
    args = parser.parse_args()

    base_url = f"http://localhost:{args.port}"

    # If a server is already running on this port, just report and exit.
    if is_server_running(base_url):
        print(f"[vLLM] A server is already running at {base_url}.")
        return 0

    proc = start_vllm_server(
        model_path=args.model,
        port=args.port,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
    )

    print(f"[vLLM] Server ready at {base_url}. Press Ctrl+C to stop.")
    try:
        proc.wait()
    except KeyboardInterrupt:
        print("\n[vLLM] Ctrl+C received; shutting down.")
        stop_vllm_server(proc)
    return proc.returncode or 0


if __name__ == "__main__":
    raise SystemExit(main())
