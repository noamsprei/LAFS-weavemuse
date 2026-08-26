"""Core sweep loop: run the WeaveMuse manager agent over every (task, variant)
pair in a dataset, capturing the full execution trace for each.

A fresh manager agent (+ fresh sub-agents/tools) is built for every pair
rather than reusing one across the sweep, because `CodeAgent.run(reset=True)`
(the default) only clears `memory.steps`, not the Python-executor's
`self.state` variable namespace -- reusing one instance would leak variables
between tasks. This is cheap: WeaveMuse's tools are lazy-loaded
(weavemuse/tools/base_tools.py::LazyLoadableTool), so building unused
sub-agents doesn't load any weights; only the shared backbone model (built
once by the caller and passed in) is expensive.
"""

from __future__ import annotations

import datetime
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from smolagents import CodeAgent

from weavemuse.agents.agents_as_tools import get_weavemuse_agents_and_tools
from weavemuse.eval.dataset import EvalTask, load_tasks
from weavemuse.eval.gpu_guard import get_free_vram_gb, release_gpu_memory
from weavemuse.eval.variants import PromptVariant, load_variants

# Short, static tool-facing description -- NOT what varies across the study.
# See weavemuse/eval/variants.py's module docstring for why description= and
# instructions= are different levers.
_MANAGER_DESCRIPTION = (
    "Manages music-related tasks: web search, music analysis, symbolic and "
    "audio music generation, and audio analysis."
)


@dataclass
class RunConfig:
    dataset_path: Path
    variants_path: Path
    output_dir: Path
    run_id: str
    tool_mode: str = "remote"
    device_map: str = "auto"
    max_steps: int = 5
    max_new_tokens: int = 1536
    model_id: str = ""
    task_ids: list[str] | None = None
    variant_ids: list[str] | None = None

    def run_dir(self) -> Path:
        return self.output_dir / self.run_id

    def trace_path(self, task_id: str, variant_id: str) -> Path:
        return self.run_dir() / "traces" / task_id / f"{variant_id}.json"


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def build_manager_agent(model, variant: PromptVariant, cfg: RunConfig) -> CodeAgent:
    """Build one fresh manager CodeAgent, with `instructions=variant.instructions`
    as the only thing that varies across the study -- mirrors
    scripts/quickstart_local.py's manager-agent construction otherwise.
    """
    weavemuse_agents, weavemuse_tools = get_weavemuse_agents_and_tools(
        model=model,
        device_map=cfg.device_map,
        tool_mode=cfg.tool_mode,
    )
    return CodeAgent(
        tools=weavemuse_tools,
        model=model,
        managed_agents=weavemuse_agents,
        name="music_manager_agent",
        description=_MANAGER_DESCRIPTION,
        instructions=variant.instructions,
        add_base_tools=True,
        max_steps=cfg.max_steps,
        additional_authorized_imports=[],
        return_full_result=True,
    )


def run_one(model, task: EvalTask, variant: PromptVariant, cfg: RunConfig) -> dict:
    """Run one (task, variant) pair and write its trace JSON. Returns the
    same record dict that gets written to disk, for manifest bookkeeping.
    """
    free_vram_before = get_free_vram_gb()

    agent = build_manager_agent(model, variant, cfg)
    try:
        result = agent.run(task.query, reset=True)
    finally:
        # Drop references before release_gpu_memory() so gc can actually
        # collect the sub-agents'/tools' tensors before empty_cache() runs.
        del agent
        release_gpu_memory()

    free_vram_after = get_free_vram_gb()

    record = {
        "task_id": task.task_id,
        "variant_id": variant.variant_id,
        "variant_instructions": variant.instructions,
        "query": task.query,
        "run_config": {
            "tool_mode": cfg.tool_mode,
            "max_steps": cfg.max_steps,
            "max_new_tokens": cfg.max_new_tokens,
            "model_id": cfg.model_id,
        },
        "output": result.output,
        "state": result.state,
        "token_usage": result.token_usage.dict() if result.token_usage else None,
        "timing": result.timing.dict(),
        "messages": result.messages,
        "free_vram_gb_before": free_vram_before,
        "free_vram_gb_after": free_vram_after,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    trace_path = cfg.trace_path(task.task_id, variant.variant_id)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)

    return record


def run_sweep(model, cfg: RunConfig) -> None:
    """Load tasks+variants, loop task x variant -> run_one, write manifest.json.

    `model` is built ONCE by the caller (e.g. scripts/run_eval.py) and reused
    across every pair -- only the manager agent around it is rebuilt per pair.
    """
    tasks = load_tasks(cfg.dataset_path)
    variants = load_variants(cfg.variants_path)

    if cfg.task_ids:
        wanted = set(cfg.task_ids)
        tasks = [t for t in tasks if t.task_id in wanted]
        missing = wanted - {t.task_id for t in tasks}
        if missing:
            raise ValueError(f"--task-ids not found in dataset: {sorted(missing)}")
    if cfg.variant_ids:
        wanted_v = set(cfg.variant_ids)
        missing_v = wanted_v - set(variants.keys())
        if missing_v:
            raise ValueError(f"--variant-ids not found in variants file: {sorted(missing_v)}")
        variants = {vid: v for vid, v in variants.items() if vid in wanted_v}

    cfg.run_dir().mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": cfg.run_id,
        "dataset_path": str(cfg.dataset_path),
        "variants_path": str(cfg.variants_path),
        "tool_mode": cfg.tool_mode,
        "max_steps": cfg.max_steps,
        "max_new_tokens": cfg.max_new_tokens,
        "model_id": cfg.model_id,
        "git_commit": _git_commit(),
        "task_ids": [t.task_id for t in tasks],
        "variant_ids": list(variants.keys()),
        "started_at": datetime.datetime.now().isoformat(),
        "runs": [],
    }

    total = len(tasks) * len(variants)
    i = 0
    for task in tasks:
        for variant in variants.values():
            i += 1
            print(f"[{i}/{total}] task={task.task_id} variant={variant.variant_id} ...")
            record = run_one(model, task, variant, cfg)
            manifest["runs"].append({
                "task_id": task.task_id,
                "variant_id": variant.variant_id,
                "state": record["state"],
                "total_tokens": (record["token_usage"] or {}).get("total_tokens"),
                "duration": record["timing"].get("duration"),
                "free_vram_gb_after": record["free_vram_gb_after"],
                "trace_path": str(cfg.trace_path(task.task_id, variant.variant_id)),
            })
            print(
                f"    state={record['state']} "
                f"free_vram_after={record['free_vram_gb_after']}"
            )

    manifest["finished_at"] = datetime.datetime.now().isoformat()
    with (cfg.run_dir() / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\nDone. Manifest: {cfg.run_dir() / 'manifest.json'}")
