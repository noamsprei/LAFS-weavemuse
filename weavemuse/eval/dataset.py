"""Loader for the eval harness's task dataset (JSONL, one task per line)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class EvalTask:
    """One task in the dataset: a single query sent to the manager agent."""

    task_id: str
    query: str
    category: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "EvalTask":
        return cls(
            task_id=d["task_id"],
            query=d["query"],
            category=d.get("category"),
            tags=list(d.get("tags", [])),
            metadata=dict(d.get("metadata", {})),
        )


def load_tasks(path: str | Path) -> list[EvalTask]:
    """Load a JSONL dataset file into a list of EvalTask.

    Each non-blank line must be a JSON object with at least "task_id" and
    "query". Raises ValueError on a duplicate task_id (silently overwriting
    tasks would make dataset bugs invisible downstream).
    """
    path = Path(path)
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
            tasks.append(task)
    return tasks
