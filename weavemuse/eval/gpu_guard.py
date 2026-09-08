"""VRAM preflight and release helpers for the eval sweep runner.

The logic here mirrors scripts/quickstart_local.py::_check_vram_headroom, but
is re-implemented independently (not imported) so that script stays untouched.

Two distinct VRAM risks motivate this module, both observed in practice
running WeaveMuse locally on an 8GB card:
  1. A stale/leftover process (e.g. a suspended `weavemuse gui`) can squat
     several GB of VRAM indefinitely -- `check_vram_headroom()` catches this
     before a sweep even starts.
  2. smolagents' CodeAgent re-sends the *entire* growing conversation every
     step (no cross-step KV-cache reuse), so VRAM usage climbs within a task
     and can ratchet up run-over-run across a dataset sweep if not released
     -- `release_gpu_memory()` is called after every (task, variant) run to
     guard against that.
"""

from __future__ import annotations

import gc
import subprocess


def check_vram_headroom(gpu_info, needed_gb: float = 5.5) -> None:
    """Bail out early with a clear diagnosis if there isn't enough free VRAM
    for the 4-bit-quantized backbone model to load entirely on-GPU.

    Why this matters: if `device_map="auto"` can't fit the model in free
    VRAM, accelerate silently offloads the remainder to CPU RAM, and every
    forward pass then shuttles activations CPU<->GPU -- this turns ~7 tok/s
    (observed on an 8GB card with the model fully on-GPU) into roughly a
    minute or two *per token*, which looks like a hang rather than an error.
    """
    if not gpu_info.has_cuda:
        return
    if gpu_info.free_vram_gb >= needed_gb:
        return

    print(
        f"\n⚠️  Only {gpu_info.free_vram_gb:.1f}GB VRAM free (~{needed_gb:.1f}GB needed for "
        f"{gpu_info.recommended_model_id} in 4-bit). Proceeding would likely force part of "
        f"the model onto CPU RAM, turning generation from ~7 tok/s into minutes per token."
    )
    try:
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        procs = ""
    if procs:
        print("Processes currently holding GPU memory:")
        print(procs)
        print("If one of these is stale (e.g. a suspended/Ctrl-Z'd run), free it with: kill -9 <pid>")
    raise SystemExit(
        "Not enough free VRAM to proceed safely -- see above. Free up VRAM and rerun, "
        "or set WEAVEMUSE_FORCE_CPU=1 to run on CPU deliberately (slow, but honest about it)."
    )


def get_free_vram_gb() -> float | None:
    """Best-effort current free-VRAM reading, for per-task manifest logging.
    Returns None if CUDA/torch isn't available rather than raising, so callers
    can log it opportunistically without special-casing CPU-only runs.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        total = torch.cuda.get_device_properties(0).total_memory
        reserved = torch.cuda.memory_reserved(0)
        return (total - reserved) / (1024**3)
    except Exception:
        return None


def release_gpu_memory() -> None:
    """Best-effort VRAM release between (task, variant) runs.

    Call this after dropping references to a run's manager agent / sub-agents
    / tools. gc.collect() first so Python actually drops the tensor refcounts
    before torch is asked to release its caching allocator's freed blocks.
    """
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
