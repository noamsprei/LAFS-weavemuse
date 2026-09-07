"""Loader for the eval harness's task dataset (JSONL, one task per line)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalTask:
    """One task in the dataset: a single query sent to the manager agent.

    `attachments` and `metadata` are deliberately separate dicts, not one:
    `attachments` is agent-visible (passed to `agent.run(additional_args=...)`
    by the runner, so its values become both literal text in the task prompt
    AND real Python variables the manager's generated code can reference --
    e.g. `{"audio_file": "..."}` makes `audio_analysis_agent(..., additional_args={"audio_file": audio_file})`
    a one-liner). Name each key after the actual tool parameter it will end
    up at (see weavemuse/eval/README.md) -- an LLM asked to "forward the
    exact path" tends to keep forwarding it under whatever variable name it
    already has rather than renaming it to the tool's real kwarg, so a
    generic key like `audio` can silently turn into a
    `TypeError: unexpected keyword argument` a couple of steps downstream.
    `metadata` is eval-only bookkeeping (e.g. `ground_truth`) that must
    NEVER reach the model -- handing the model its own answer key defeats
    the point of the eval. Keep any reference/expected-answer field in
    `metadata`, and only file paths the agent should actually use in
    `attachments`.
    """

    task_id: str
    query: str
    category: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    attachments: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalTask":
        return cls(
            task_id=d["task_id"],
            query=d["query"],
            category=d.get("category"),
            tags=list(d.get("tags", [])),
            metadata=dict(d.get("metadata", {})),
            attachments=dict(d.get("attachments", {})),
        )


def _resolve_attachments(task: EvalTask, base_dir: Path, path: Path, line_no: int) -> None:
    """Resolve every attachment path relative to the dataset file's own
    directory (not CWD, which may be the repo root when `--dataset` is a
    relative path elsewhere) into an absolute path, in place. Raises
    ValueError naming the task/key/path on a missing file -- a silently
    unresolved attachment would surface later as a confusing tool-side
    "file not found" deep in a trace instead of a clear load-time error.
    """
    for key, value in task.attachments.items():
        if not isinstance(value, str):
            continue
        resolved = (base_dir / value).resolve()
        if not resolved.is_file():
            raise ValueError(
                f"{path}:{line_no}: task {task.task_id!r} attachments[{key!r}] "
                f"= {value!r} does not resolve to an existing file "
                f"(looked for {resolved})"
            )
        task.attachments[key] = str(resolved)


def load_tasks(path: str | Path) -> list[EvalTask]:
    """Load a JSONL dataset file into a list of EvalTask.

    Each non-blank line must be a JSON object with at least "task_id" and
    "query". Raises ValueError on a duplicate task_id (silently overwriting
    tasks would make dataset bugs invisible downstream), or on an
    "attachments" path that doesn't resolve to an existing file.
    """
    path = Path(path)
    base_dir = path.parent
    tasks: list[EvalTask] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {e}") from e
            task = EvalTask.from_dict(d)
            if task.task_id in seen_ids:
                raise ValueError(f"{path}:{line_no}: duplicate task_id {task.task_id!r}")
            seen_ids.add(task.task_id)
            _resolve_attachments(task, base_dir, path, line_no)
            tasks.append(task)
    return tasks
