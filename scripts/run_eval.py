"""CLI: run the WeaveMuse manager agent over a dataset of tasks under one or
more prompt variants, capturing the full execution trace of each run.

Part of the trace-quality evaluation harness (weavemuse/eval/). See
weavemuse/eval/README.md for the full guide -- setup, how to write your own
dataset/variants, what to watch for.

Setup:
    1. Fill in HF_TOKEN in .env (needed for the remote NotaGen/StableAudio/
       AudioFlamingo tools used under the default --tool-mode remote).
    2. uv sync --extra gpu --extra-index-url https://download.pytorch.org/whl/cu121
    3. Run:
        .venv/bin/python scripts/run_eval.py \\
            --dataset data/eval/tasks_smoke.jsonl \\
            --variants data/eval/variants_smoke.json \\
            --run-id smoke_test

Backbone LLM runs LOCALLY (same VRAM-tier auto-selection as
scripts/quickstart_local.py) -- by design, per this study's premise that a
weaker local model shows more visible prompt-sensitivity than a strong cloud
model would. --tool-mode defaults to "remote" (not "hybrid") specifically to
avoid the LARGE local music models (StableAudio ~5GB, ChatMusician ~4-8GB) so
they don't compound the per-task VRAM growth that's already a real
constraint on an 8GB card (see gpu_guard.py's docstring). Note this isn't
literally zero local GPU use even in --tool-mode remote: RemoteNotaGenTool
still loads a small (~516M param) local NotaGen model despite its name --
confirmed in practice, this cost real VRAM headroom (down to ~1.2GB free
after a single task on this 8GB card) and contributed to an observed CUDA
OOM during development (now caught and recorded as state="error" rather than
crashing the sweep -- see runner.py::run_one()).
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = REPO_ROOT / "data" / "eval" / "tasks_smoke.jsonl"
DEFAULT_VARIANTS = REPO_ROOT / "data" / "eval" / "variants_smoke.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "eval"


def _require_env(var_name: str) -> None:
    if not os.getenv(var_name):
        raise SystemExit(
            f"{var_name} is not set.\n"
            f"Fill in {var_name} in .env, or set it in your shell before running this script."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the WeaveMuse manager agent over a dataset x prompt-variant sweep."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET,
                         help=f"JSONL task file (default: {DEFAULT_DATASET}).")
    parser.add_argument("--variants", type=Path, default=DEFAULT_VARIANTS,
                         help=f"JSON prompt-variants file (default: {DEFAULT_VARIANTS}).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                         help=f"Directory to write outputs/eval/<run_id>/ into (default: {DEFAULT_OUTPUT_DIR}).")
    parser.add_argument("--run-id", required=True,
                         help="Name for this sweep's output subdirectory, e.g. smoke_test.")
    parser.add_argument("--tool-mode", choices=["remote", "hybrid"], default="remote",
                         help="'remote' (default, recommended): avoids the large local music "
                              "models (StableAudio, ChatMusician); RemoteNotaGenTool still loads "
                              "a small local model despite its name (~1GB observed). 'hybrid': "
                              "also load the large local music tools -- only if your dataset "
                              "specifically needs them; compounds VRAM pressure further.")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--task-ids", default=None,
                         help="Comma-separated subset of task_ids to run (default: all).")
    parser.add_argument("--variant-ids", default=None,
                         help="Comma-separated subset of variant_ids to run (default: all).")
    parser.add_argument("--limit", type=int, default=None,
                         help="Cap the dataset to the first N tasks (after --task-ids filtering).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_env("HF_TOKEN")

    from weavemuse.agents.models import TransformersModel
    from weavemuse.eval.gpu_guard import check_vram_headroom
    from weavemuse.eval.runner import RunConfig, run_sweep
    from weavemuse.utils.gpu_utils import (
        check_force_cpu_mode,
        detect_gpu_capabilities,
        get_quantization_config,
        print_gpu_summary,
    )

    gpu_info = detect_gpu_capabilities(force_cpu=check_force_cpu_mode())
    print_gpu_summary(gpu_info)
    check_vram_headroom(gpu_info)

    quantization_config = get_quantization_config(gpu_info)
    print("✅ Using 4-bit quantization for optimal VRAM usage" if quantization_config
          else "⚠️  Quantization disabled (CPU mode or not available)")

    print(f"Backbone model: {gpu_info.recommended_model_id} (local, device_map={gpu_info.device_map})")
    model = TransformersModel(
        model_id=gpu_info.recommended_model_id,
        trust_remote_code=True,
        device_map=gpu_info.device_map,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        offload_buffers=True,
        quantization_config=quantization_config,
        max_new_tokens=args.max_new_tokens,
    )

    cfg = RunConfig(
        dataset_path=args.dataset,
        variants_path=args.variants,
        output_dir=args.output_dir,
        run_id=args.run_id,
        tool_mode=args.tool_mode,
        device_map=gpu_info.device_map,
        max_steps=args.max_steps,
        max_new_tokens=args.max_new_tokens,
        model_id=gpu_info.recommended_model_id,
        task_ids=args.task_ids.split(",") if args.task_ids else None,
        variant_ids=args.variant_ids.split(",") if args.variant_ids else None,
    )

    if args.limit is not None:
        from weavemuse.eval.dataset import load_tasks
        all_ids = [t.task_id for t in load_tasks(cfg.dataset_path)]
        limited = (cfg.task_ids or all_ids)[: args.limit]
        cfg.task_ids = limited

    run_sweep(model, cfg)


if __name__ == "__main__":
    main()
