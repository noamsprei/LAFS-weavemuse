"""Per-tool-call and per-sub-agent trace capture, for evaluation.

Today's trace (weavemuse/eval/runner.py's `record["messages"]`, i.e. the
manager CodeAgent's own `memory.get_full_steps()`) only sees the manager's
own steps. For a CodeAgent, each step's `tool_calls` is always a single
synthetic entry (`name="python_interpreter"`, the whole generated code
blob) -- never the individual tool/sub-agent calls made *inside* that code.
Worse, a managed sub-agent's own Thought/Code/Observation cycle (e.g.
`audio_analysis_agent` trying `audio_flamingo` first, then falling back to
`audio_analysis` per its own prompt instruction) is discarded entirely --
`MultiStepAgent.__call__` only returns the sub-agent's bare final-answer
text unless `provide_run_summary=True` (never set by
weavemuse/agents/agents_as_tools.py), so only that flattened string reaches
the manager's own `observations` field. Confirmed empirically: reading a
real s001 trace showed no visibility into what `audio_analysis_agent`
actually tried internally.

Two hook points cover this, since a CodeAgent's Python executor injects both
`self.tools` (smolagents Tool instances) and `self.managed_agents` (plain
CodeAgent instances) into the same executor namespace and the LLM's
generated code just calls them as ordinary Python functions:
  - `Tool.__call__` -- the single dispatcher every tool call passes through
    before `forward()` runs, at every nesting level (manager and every
    sub-agent alike).
  - `MultiStepAgent.__call__` -- the parallel entry point for a manager
    calling a managed sub-agent (sub-agents are plain CodeAgent instances,
    not Tool subclasses; CodeAgent never overrides `__call__`).

Both are monkeypatched only for the duration of one `agent.run()` call (see
`traced_run()`), then restored -- pure observation, no interception or
behavior change of any kind. Safe only because weavemuse/eval's own sweep
loop (runner.py::run_sweep) is strictly sequential/single-threaded -- do not
reuse this context manager concurrently across threads.
"""

from __future__ import annotations

import contextlib
import datetime
import time
from dataclasses import dataclass, field
from typing import Any

from smolagents.agents import MultiStepAgent
from smolagents.tools import Tool

_TRUNCATE_LIMIT = 800


def _truncate(value: Any, limit: int = _TRUNCATE_LIMIT) -> str:
    s = value if isinstance(value, str) else repr(value)
    if len(s) > limit:
        return s[:limit] + f"... [truncated, {len(s)} chars total]"
    return s


@dataclass
class LedgerEntry:
    seq: int
    kind: str  # "tool" | "managed_agent"
    name: str
    args: str  # truncated repr of the call's args/kwargs
    action: str  # "called" | "error"
    output: str | None  # truncated str(result), or the exception text
    start_ts: str
    duration_s: float


@dataclass
class ToolLedger:
    """One fresh instance per `traced_run()` call -- never accumulates
    across a sweep (each run_one() call gets its own ledger)."""

    entries: list[LedgerEntry] = field(default_factory=list)
    _seq: int = 0

    def record(self, **kwargs) -> None:
        self._seq += 1
        self.entries.append(LedgerEntry(seq=self._seq, **kwargs))

    def as_list(self) -> list[dict]:
        return [e.__dict__ for e in self.entries]


@contextlib.contextmanager
def traced_run(agent):
    """with traced_run(agent) as ledger:
           result = agent.run(...)
       ledger.as_list() -- the ordered per-call trace for this run only,
       stable to read even if agent.run() raised (the ledger is bound as
       soon as this context manager starts, so it holds every call
       recorded before any later failure).
    """
    ledger = ToolLedger()

    orig_tool_call = Tool.__call__
    orig_agent_call = MultiStepAgent.__call__

    def patched_tool_call(self, *args, **kwargs):
        start = time.time()
        start_ts = datetime.datetime.now().isoformat()
        try:
            output = orig_tool_call(self, *args, **kwargs)
        except Exception as e:
            ledger.record(
                kind="tool", name=self.name,
                args=_truncate({**{f"arg{i}": a for i, a in enumerate(args)}, **kwargs}),
                action="error", output=f"{type(e).__name__}: {e}",
                start_ts=start_ts, duration_s=time.time() - start,
            )
            raise
        ledger.record(
            kind="tool", name=self.name,
            args=_truncate({**{f"arg{i}": a for i, a in enumerate(args)}, **kwargs}),
            action="called", output=_truncate(output),
            start_ts=start_ts, duration_s=time.time() - start,
        )
        return output

    def patched_agent_call(self, task, **kwargs):
        start = time.time()
        start_ts = datetime.datetime.now().isoformat()
        try:
            output = orig_agent_call(self, task, **kwargs)
        except Exception as e:
            ledger.record(
                kind="managed_agent", name=self.name,
                args=_truncate({"task": task, **kwargs}),
                action="error", output=f"{type(e).__name__}: {e}",
                start_ts=start_ts, duration_s=time.time() - start,
            )
            raise
        ledger.record(
            kind="managed_agent", name=self.name,
            args=_truncate({"task": task, **kwargs}),
            action="called", output=_truncate(output),
            start_ts=start_ts, duration_s=time.time() - start,
        )
        return output

    Tool.__call__ = patched_tool_call
    MultiStepAgent.__call__ = patched_agent_call
    try:
        yield ledger
    finally:
        Tool.__call__ = orig_tool_call
        MultiStepAgent.__call__ = orig_agent_call


def collect_sub_agent_traces(agent) -> dict[str, list[dict]]:
    """Call right after agent.run() returns, before `del agent` -- the
    manager retains live references to every sub-agent via
    `agent.managed_agents` ({name: agent}) for the lifetime of the run.

    `sub_agent.memory.get_full_steps()` already returns plain JSON-ready
    dicts (the same shape `result.messages` already is), so this recovers
    each sub-agent's own Thought/Code/Observation cycle -- previously
    invisible in the manager's own trace.

    Caveat, documented rather than hidden: invoking a managed agent calls
    `self.run(full_task, reset=True)` (smolagents default), which clears
    `memory.steps` at the start of each call -- so if a given sub-agent is
    invoked more than once within one task, only its LAST invocation's
    steps survive here, not a full history of every call. Acceptable given
    these sub-agents cap at max_steps=1 or 2 and are typically invoked once
    per task in practice.
    """
    return {
        name: sub_agent.memory.get_full_steps()
        for name, sub_agent in agent.managed_agents.items()
    }
