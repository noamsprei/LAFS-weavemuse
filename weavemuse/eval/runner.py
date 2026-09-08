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
from weavemuse.eval.tool_tracing import collect_sub_agent_traces, traced_run
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
    max_new_tokens: int = 4096  # 1536 truncated sub-agent code blocks -> parse-error loops
    model_id: str = ""
    task_ids: list[str] | None = None
    variant_ids: list[str] | None = None
    exclude_agents: list[str] | None = None
    overwrite: bool = False  # redo (task, variant) pairs whose trace already exists
    # JSON file of {question_template -> method block}. Consumed only by
    # variants with query_mode="expert"; None disables expert-query composition.
    expert_prompts_path: Path | None = None

    def run_dir(self) -> Path:
        return self.output_dir / self.run_id

    def trace_path(self, task_id: str, variant_id: str) -> Path:
        return self.run_dir() / "traces" / task_id / f"{variant_id}.json"


def _completed_trace(path: Path) -> dict | None:
    """Return a previously-written trace if it exists and parses (a resume
    checkpoint), else None so the pair is (re)run."""
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            rec = json.load(f)
        return rec if rec.get("state") else None
    except (json.JSONDecodeError, OSError):
        return None  # partial/corrupt write -> rerun


def _manifest_row(cfg: RunConfig, rec: dict, reused: bool) -> dict:
    return {
        "task_id": rec["task_id"],
        "variant_id": rec["variant_id"],
        "query_mode": rec.get("query_mode"),
        "state": rec["state"],
        "reused": reused,
        "total_tokens": (rec.get("token_usage") or {}).get("total_tokens"),
        "duration": (rec.get("timing") or {}).get("duration"),
        "error": rec.get("error"),
        "free_vram_gb_after": rec.get("free_vram_gb_after"),
        "trace_path": str(cfg.trace_path(rec["task_id"], rec["variant_id"])),
    }


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def load_expert_prompts(path: Path | None) -> dict[str, str]:
    """Load the {question_template -> method block} JSON. Returns {} when path
    is None or the file is absent, so a dataset with no expert prompts still
    runs (expert-mode variants then just fall back to the base query, with a
    per-pair warning from _compose_query)."""
    if path is None or not Path(path).is_file():
        return {}
    with Path(path).open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a JSON object of template -> block string")
    return {k: v for k, v in raw.items()}


def _compose_query(task: EvalTask, variant: PromptVariant, expert_prompts: dict[str, str]) -> str:
    """The effective query for one (task, variant) pair. query_mode="base" (or
    an expert-mode variant with no matching block) returns the task's query
    verbatim; query_mode="expert" appends the method block keyed by the task's
    `category` (its question template)."""
    if variant.query_mode != "expert":
        return task.query
    block = expert_prompts.get(task.category or "")
    if not block:
        print(f"    ⚠️  variant {variant.variant_id!r} is query_mode=expert but no "
              f"expert block for category {task.category!r} -- sending base query "
              f"(this makes it identical to the base condition for this task).")
        return task.query
    return task.query + "\n\n" + block


def build_manager_agent(model, variant: PromptVariant, cfg: RunConfig) -> CodeAgent:
    """Build one fresh manager CodeAgent with `instructions=variant.instructions`
    -- mirrors scripts/quickstart_local.py's manager-agent construction
    otherwise. For the musicology study `instructions` is held identical across
    variants and the intervention rides in the query instead (variant.query_mode
    / _compose_query); older studies vary `instructions` here.
    """
    weavemuse_agents, weavemuse_tools = get_weavemuse_agents_and_tools(
        model=model,
        device_map=cfg.device_map,
        tool_mode=cfg.tool_mode,
        exclude_agents=cfg.exclude_agents,
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
        # the prompts tell the agent tool results are JSON to parse; give it json
        # (+ the light stdlib the sub-agent already has, for aggregating results)
        additional_authorized_imports=["json", "math", "statistics", "collections", "re"],
        return_full_result=True,
    )


def run_one(
    model,
    task: EvalTask,
    variant: PromptVariant,
    cfg: RunConfig,
    expert_prompts: dict[str, str] | None = None,
) -> dict:
    """Run one (task, variant) pair and write its trace JSON. Returns the
    same record dict that gets written to disk, for manifest bookkeeping.

    Any exception from agent.run() (observed in practice: a CUDA OOM inside a
    nested sub-agent's own generate() call, which smolagents does NOT always
    absorb internally -- it can propagate all the way out of .run()) is
    caught here rather than left to crash the whole sweep. One bad
    (task, variant) pair records state="error" and the sweep moves on,
    instead of silently losing every pair after it and never writing
    manifest.json.
    """
    free_vram_before = get_free_vram_gb()
    effective_query = _compose_query(task, variant, expert_prompts or {})

    agent = None
    result = None
    error_text = None
    ledger = None
    try:
        agent = build_manager_agent(model, variant, cfg)
        # attachments (e.g. {"audio_file": "/abs/path.wav"}) ride in via
        # smolagents' own additional_args mechanism: merged into the
        # manager's Python-executor state (so its generated code can
        # reference `audio_file` as a real variable) AND appended to the
        # task text -- NOT task.metadata, which may hold eval-only fields
        # like ground_truth that must never reach the model.
        #
        # traced_run() captures every individual tool/sub-agent call (not
        # just the manager's own flattened step text) for the trace record
        # below -- see tool_tracing.py's module docstring. `ledger` is bound
        # as soon as the `with` block starts, before agent.run() runs, so it
        # still holds every call recorded even if agent.run() itself raises.
        with traced_run(agent) as ledger:
            result = agent.run(
                effective_query, reset=True, additional_args=task.attachments or None
            )
    except Exception as e:
        error_text = f"{type(e).__name__}: {e}"
        print(f"    ⚠️  run failed: {error_text}")
    finally:
        tool_calls = ledger.as_list() if ledger is not None else []
        sub_agent_traces = collect_sub_agent_traces(agent) if agent is not None else {}
        # Drop references before release_gpu_memory() so gc can actually
        # collect the sub-agents'/tools' tensors before empty_cache() runs.
        del agent
        release_gpu_memory()

    free_vram_after = get_free_vram_gb()

    if result is not None:
        output = result.output
        state = result.state
        token_usage = result.token_usage.dict() if result.token_usage else None
        timing = result.timing.dict()
        messages = result.messages
    else:
        output, state, token_usage, timing, messages = None, "error", None, None, []

    record = {
        "task_id": task.task_id,
        "variant_id": variant.variant_id,
        "variant_instructions": variant.instructions,
        "query_mode": variant.query_mode,
        # `query` is what the agent actually received (base + expert block when
        # query_mode=expert); `base_query` is the task's query verbatim, so a
        # judge can be shown the base question regardless of condition.
        "query": effective_query,
        "base_query": task.query,
        "question_template": task.category,
        "attachments": task.attachments,
        "run_config": {
            "tool_mode": cfg.tool_mode,
            "max_steps": cfg.max_steps,
            "max_new_tokens": cfg.max_new_tokens,
            "model_id": cfg.model_id,
            "exclude_agents": cfg.exclude_agents,
        },
        "output": output,
        "state": state,
        "error": error_text,
        "token_usage": token_usage,
        "timing": timing,
        "messages": messages,
        "tool_calls": tool_calls,
        "sub_agent_traces": sub_agent_traces,
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
    expert_prompts = load_expert_prompts(cfg.expert_prompts_path)

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

    # Validate expert-query wiring against the variants/tasks actually selected.
    if any(v.query_mode == "expert" for v in variants.values()):
        if not expert_prompts:
            raise ValueError(
                f"a selected variant uses query_mode='expert' but no expert-prompts "
                f"file was loaded (expert_prompts_path={cfg.expert_prompts_path!r}). "
                f"Pass --expert-prompts pointing at a template->block JSON file."
            )
        templates = {k for k in expert_prompts if not k.startswith("_")}
        missing_blocks = {t.category for t in tasks if t.category} - templates
        if missing_blocks:
            print(f"⚠️  expert-prompts file has no block for these question "
                  f"templates: {sorted(missing_blocks)} -- expert runs for those "
                  f"tasks will fall back to the base query.")

    cfg.run_dir().mkdir(parents=True, exist_ok=True)
    manifest_path = cfg.run_dir() / "manifest.json"

    manifest = {
        "run_id": cfg.run_id,
        "dataset_path": str(cfg.dataset_path),
        "variants_path": str(cfg.variants_path),
        "expert_prompts_path": str(cfg.expert_prompts_path) if cfg.expert_prompts_path else None,
        "query_modes": {vid: v.query_mode for vid, v in variants.items()},
        "tool_mode": cfg.tool_mode,
        "max_steps": cfg.max_steps,
        "max_new_tokens": cfg.max_new_tokens,
        "model_id": cfg.model_id,
        "exclude_agents": cfg.exclude_agents,
        "overwrite": cfg.overwrite,
        "git_commit": _git_commit(),
        "task_ids": [t.task_id for t in tasks],
        "variant_ids": list(variants.keys()),
        "started_at": datetime.datetime.now().isoformat(),
        "runs": [],
    }

    def _flush_manifest() -> None:
        manifest["updated_at"] = datetime.datetime.now().isoformat()
        tmp = manifest_path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)
        tmp.replace(manifest_path)  # atomic: a crash mid-write never truncates it

    total = len(tasks) * len(variants)
    i = n_run = n_reused = 0
    for task in tasks:
        for variant in variants.values():
            i += 1
            done = None if cfg.overwrite else _completed_trace(
                cfg.trace_path(task.task_id, variant.variant_id)
            )
            if done is not None:
                n_reused += 1
                print(f"[{i}/{total}] task={task.task_id} variant={variant.variant_id} "
                      f"-- reusing existing trace (state={done['state']})")
                manifest["runs"].append(_manifest_row(cfg, done, reused=True))
                _flush_manifest()
                continue

            n_run += 1
            print(f"[{i}/{total}] task={task.task_id} variant={variant.variant_id} ...")
            record = run_one(model, task, variant, cfg, expert_prompts)  # writes its trace to disk
            manifest["runs"].append(_manifest_row(cfg, record, reused=False))
            _flush_manifest()  # checkpoint after every pair -- a crash keeps all prior work
            print(f"    state={record['state']} "
                  f"free_vram_after={record['free_vram_gb_after']}")

    manifest["finished_at"] = datetime.datetime.now().isoformat()
    _flush_manifest()
    print(f"\nDone. {n_run} run, {n_reused} reused. Manifest: {manifest_path}")
