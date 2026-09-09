"""CLI: flatten eval trace JSONs (from scripts/run_eval.py) into per-step CSVs.

A trace JSON is deeply nested and hard to scan by eye: the manager agent's own
steps (record["messages"]) mostly just say "delegate to sub-agent X" / "here's
its flattened final answer" -- the actual tool calls (get_tonal_plan,
get_harmony, ...) and their raw answers live inside that sub-agent's own steps
(record["sub_agent_traces"][X]), invisible unless you also read those. This
script interleaves the manager's steps with every sub-agent's steps into one
chronological (by each step's own start time) CSV per trace, one row per step,
with columns: which agent acted, which tool it called, whether the step
errored, the tool's raw output, and the agent's own "Thought:" reasoning for
that step (what it inferred from the prior observation and what it did next).

Usage:
    .venv/bin/python scripts/trace_to_csv.py outputs/eval/sw1_modulation__0012/expert.json
    .venv/bin/python scripts/trace_to_csv.py outputs/eval/<run_id>/traces --out-dir /tmp/csvs
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

# Identifiers that show up in generated code but are never a WeaveMuse tool/
# sub-agent call -- filtered out of the tool_called guess below.
_NOT_A_TOOL_CALL = {
    "print", "len", "str", "int", "float", "dict", "list", "set", "tuple",
    "range", "enumerate", "zip", "sorted", "isinstance", "type", "open",
    "json.loads", "json.dumps", "round", "abs", "min", "max", "sum", "any",
    "all", "repr", "format",
}
_CALL_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_.]*)\(")


def _truncate(value: Any, limit: int) -> str:
    s = value if isinstance(value, str) else json.dumps(value, default=str)
    if len(s) > limit:
        return s[:limit] + f"... [truncated, {len(s)} chars total]"
    return s


def _guess_tool_called(code_action: str | None) -> str:
    """Best-effort: the LLM's system prompt tells it not to chain multiple
    tool calls in one code block, so in practice each step calls exactly one
    tool/sub-agent function. Report every non-builtin call found, in case
    that's ever violated, rather than silently picking the first."""
    if not code_action:
        return ""
    names = dict.fromkeys(  # de-dupe, keep first-seen order
        m for m in _CALL_RE.findall(code_action) if m not in _NOT_A_TOOL_CALL
    )
    return "; ".join(names)


def _clean_observation(observations: str | None) -> str:
    if not observations:
        return ""
    s = observations
    s = s.removeprefix("Execution logs:\n")
    s = s.removesuffix("\nLast output from code snippet:\nNone")
    return s.strip()


def _extract_reasoning(model_output: str | None) -> str:
    """model_output is "Thought: ...\\n\\nCode:\\n```python\\n...\\n```" (or the
    older "<code>...</code>" delimiter). Keep only the Thought part -- the
    code itself is already in the 'code' column, no need to duplicate it."""
    if not model_output:
        return ""
    for marker in ("\nCode:\n```", "\n<code>", "\n```py"):
        idx = model_output.find(marker)
        if idx != -1:
            return model_output[:idx].strip()
    return model_output.strip()


def _step_row(agent: str, step: dict, max_chars: int) -> dict:
    error = step.get("error")
    return {
        "agent": agent,
        "agent_step_number": step.get("step_number"),
        "tool_called": _guess_tool_called(step.get("code_action")),
        "status": "error" if error else "success",
        "tool_output": _truncate(_clean_observation(step.get("observations")), max_chars),
        "reasoning": _truncate(_extract_reasoning(step.get("model_output")), max_chars),
        "code": _truncate(step.get("code_action") or "", max_chars),
        "error": _truncate(error, max_chars) if error else "",
        "is_final_answer": bool(step.get("is_final_answer")),
        "duration_s": (step.get("timing") or {}).get("duration"),
        "_start_time": (step.get("timing") or {}).get("start_time"),
    }


_FIELDNAMES = [
    "task_id", "variant_id", "question_template",
    "global_step", "agent", "agent_step_number", "tool_called", "status",
    "tool_output", "reasoning", "code", "error", "is_final_answer", "duration_s",
]


def trace_to_rows(record: dict, max_chars: int) -> list[dict]:
    """One row per step, manager and every sub-agent interleaved by each
    step's own start_time (a sub-agent's steps naturally fall between the
    manager step that invoked it and the manager step that reads its result,
    since the manager step's own timing spans the whole nested call)."""
    rows: list[dict] = []

    # messages[0] is the TaskStep (no step_number) -- skip it.
    for step in (record.get("messages") or [])[1:]:
        rows.append(_step_row("manager", step, max_chars))
    for agent_name, steps in (record.get("sub_agent_traces") or {}).items():
        for step in (steps or [])[1:]:
            rows.append(_step_row(agent_name, step, max_chars))

    # Fall back to insertion order (manager first, then sub-agents in dict
    # order) for any row whose start_time is missing, rather than crashing --
    # an "error"-state trace can have partial/absent timing.
    rows.sort(key=lambda r: (r["_start_time"] is None, r["_start_time"]))

    for i, row in enumerate(rows, start=1):
        row["global_step"] = i
        row["task_id"] = record.get("task_id")
        row["variant_id"] = record.get("variant_id")
        row["question_template"] = record.get("question_template")
        del row["_start_time"]

    return rows


def convert_file(src: Path, dest: Path, max_chars: int) -> int:
    with src.open("r", encoding="utf-8") as f:
        record = json.load(f)

    rows = trace_to_rows(record, max_chars)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Flatten eval trace JSON(s) into per-step CSVs (manager + "
                    "sub-agent steps interleaved chronologically)."
    )
    parser.add_argument("path", type=Path,
                         help="A single trace .json file, or a directory to search "
                              "recursively for trace JSONs (e.g. outputs/eval/<run_id>/traces).")
    parser.add_argument("--out-dir", type=Path, default=None,
                         help="Write CSVs here, mirroring --path's relative directory "
                              "structure, instead of next to each source .json.")
    parser.add_argument("--max-chars", type=int, default=1000,
                         help="Truncate tool_output/reasoning/code fields beyond this "
                              "many characters (default: 1000).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.path.is_file():
        src_paths = [args.path]
        base_dir = args.path.parent
    elif args.path.is_dir():
        # Same exclusions as scripts/run_judge.py: manifest.json and
        # *.judge.json sit alongside trace JSONs under a run dir but aren't
        # trace files themselves.
        src_paths = sorted(
            p for p in args.path.rglob("*.json")
            if p.name != "manifest.json" and not p.name.endswith(".judge.json")
        )
        base_dir = args.path
        if not src_paths:
            raise SystemExit(f"No trace JSON files found under {args.path}")
    else:
        raise SystemExit(f"{args.path} does not exist")

    n_ok = n_skipped = 0
    for i, src in enumerate(src_paths, start=1):
        if args.out_dir:
            dest = (args.out_dir / src.relative_to(base_dir)).with_suffix(".csv")
        else:
            dest = src.with_suffix(".csv")
        print(f"[{i}/{len(src_paths)}] {src} -> {dest} ...", end=" ")
        try:
            n_rows = convert_file(src, dest, args.max_chars)
        except (KeyError, json.JSONDecodeError) as e:
            print(f"⚠️  skipped: not a valid trace file ({type(e).__name__}: {e})")
            n_skipped += 1
            continue
        print(f"{n_rows} rows")
        n_ok += 1

    print(f"\nDone. {n_ok} file(s) converted, {n_skipped} skipped.")


if __name__ == "__main__":
    main()
