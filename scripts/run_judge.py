"""CLI: score saved trace JSONs (from scripts/run_eval.py) with an LLM judge.

Deliberately a fully SEPARATE process/invocation from run_eval.py: this means
phase 1's GPU memory (the backbone LLM) is completely released on process
exit before phase 2 ever allocates anything, even if --backend local also
loads a model.

Setup: same .env as scripts/run_eval.py. --backend remote additionally needs
ANTHROPIC_API_KEY set (in .env or your shell).

Usage:
    .venv/bin/python scripts/run_judge.py \\
        --traces-dir outputs/eval/smoke_test/traces \\
        --backend local \\
        --limit 2
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUBRIC = REPO_ROOT / "weavemuse" / "eval" / "rubrics" / "default.json"


def _require_env(var_name: str) -> None:
    if not os.getenv(var_name):
        raise SystemExit(
            f"{var_name} is not set.\n"
            f"Fill in {var_name} in .env, or set it in your shell before running this script."
        )


def _warn_missing_env(var_name: str, why: str) -> None:
    if not os.getenv(var_name):
        print(f"⚠️  {var_name} not set -- {why}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score saved eval traces with an LLM judge.")
    parser.add_argument("--traces-dir", type=Path, required=True,
                         help="Directory of trace JSONs (e.g. outputs/eval/<run_id>/traces), "
                              "searched recursively for *.json files.")
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Where to write <task_id>/<variant_id>.judge.json files "
                              "(default: sibling 'scores/' dir next to --traces-dir).")
    parser.add_argument("--rubric", type=Path, default=DEFAULT_RUBRIC)
    parser.add_argument("--dataset", type=Path, default=None,
                         help="The tasks JSONL used for the run. If given, each task's "
                              "metadata.reference (a domain-expert answer) is shown to the "
                              "judge to score task_success against.")
    parser.add_argument("--backend", choices=["local", "remote"], default="local",
                         help="'local' (default): zero API cost, reuses a local TransformersModel "
                              "-- but see judge.py's self-judging-bias caveat if it's the same "
                              "model that generated the traces. 'remote': Anthropic API, costs "
                              "real tokens -- always start with a tight --limit.")
    parser.add_argument("--judge-model-id", default=None,
                         help="Backend-specific model id override. remote default: claude-haiku-4-5 "
                              "(cheapest current Claude model). local default: same VRAM-tier "
                              "auto-selection as scripts/run_eval.py.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Score only the first N trace files -- ALWAYS use this for a first "
                              "--backend remote run to bound cost before scoring a whole dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    from weavemuse.eval.judge import score_trace_file

    with args.rubric.open("r", encoding="utf-8") as f:
        rubric = json.load(f)

    trace_paths = sorted(args.traces_dir.rglob("*.json"))
    if args.limit is not None:
        trace_paths = trace_paths[: args.limit]
    if not trace_paths:
        raise SystemExit(f"No trace JSON files found under {args.traces_dir}")

    tasks_by_id = None
    if args.dataset:
        tasks_by_id = {}
        for line in args.dataset.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                d = json.loads(line)
                tasks_by_id[d["task_id"]] = d
        n_ref = sum(1 for d in tasks_by_id.values() if (d.get("metadata") or {}).get("reference"))
        print(f"Loaded {len(tasks_by_id)} tasks ({n_ref} with a reference answer for the judge).")

    output_dir = args.output_dir or (args.traces_dir.parent / "scores")

    if args.backend == "remote":
        _require_env("ANTHROPIC_API_KEY")
        from weavemuse.eval.judge import RemoteJudgeBackend

        judge_model_id = args.judge_model_id or "claude-haiku-4-5"
        backend = RemoteJudgeBackend(model_id=judge_model_id)
        print(f"⚠️  --backend remote: {len(trace_paths)} calls to {judge_model_id}. "
              f"Re-run with --limit if this isn't what you meant.")
    else:
        _warn_missing_env("HF_TOKEN", "only needed for gated model downloads; "
                          "the local judge model is ungated.")
        from weavemuse.agents.models import TransformersModel
        from weavemuse.eval.gpu_guard import check_vram_headroom
        from weavemuse.eval.judge import LocalJudgeBackend
        from weavemuse.utils.gpu_utils import (
            check_force_cpu_mode,
            detect_gpu_capabilities,
            get_quantization_config,
            print_gpu_summary,
        )

        gpu_info = detect_gpu_capabilities(force_cpu=check_force_cpu_mode())
        print_gpu_summary(gpu_info)
        check_vram_headroom(gpu_info)
        judge_model_id = args.judge_model_id or gpu_info.recommended_model_id
        quantization_config = get_quantization_config(gpu_info)
        model = TransformersModel(
            model_id=judge_model_id,
            trust_remote_code=True,
            device_map=gpu_info.device_map,
            torch_dtype="auto",
            low_cpu_mem_usage=True,
            offload_buffers=True,
            quantization_config=quantization_config,
            max_new_tokens=512,
        )
        backend = LocalJudgeBackend(model)
        print("Note: --backend local reuses the same weak backbone model class as run_eval.py -- "
              "if it's the same model that generated these traces, expect shared blind spots.")

    for i, trace_path in enumerate(trace_paths, start=1):
        print(f"[{i}/{len(trace_paths)}] scoring {trace_path} ...")
        record = score_trace_file(
            trace_path, backend, rubric,
            judge_backend_name=args.backend, judge_model_id=judge_model_id,
            tasks_by_id=tasks_by_id,
        )
        out_path = output_dir / record["task_id"] / f"{record['variant_id']}.judge.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, default=str)
        if record.get("parse_error"):
            print(f"    ⚠️  parse_error: {record['parse_error']}")
        else:
            print(f"    scores: { {k: v.get('score') for k, v in record['scores'].items()} }")

    print(f"\nDone. Scores written under {output_dir}")


if __name__ == "__main__":
    main()
