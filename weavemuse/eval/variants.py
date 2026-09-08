"""Loader for prompt variants: named `instructions` strings passed to the
manager `CodeAgent`'s `instructions=` constructor kwarg.

`instructions` (not `description`) is what actually changes an agent's own
reasoning: it gets woven into that agent's system prompt template as
`custom_instructions` (see smolagents.agents.CodeAgent.initialize_system_prompt).
`description` only affects how a *parent* agent's prompt describes this agent
as a callable tool -- irrelevant here since the manager is the top-level agent.

`query_mode` is the second lever, added for the musicology study: instead of
(or as well as) varying `instructions`, a variant can select how the per-task
query is built. "base" sends the task's `query` as-is; "expert" appends the
per-question-template decomposition block from an expert-prompts file (see
runner.py::_compose_query and data/eval/expert_prompts_musicology.json). This
keeps the domain intervention in the question, per-question-type, rather than
in a single fixed manager-prompt prefix.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VARIANT_ID = "default"


QUERY_MODES = ("base", "expert")


@dataclass
class PromptVariant:
    variant_id: str
    instructions: str
    description: str = ""
    query_mode: str = "base"

    @classmethod
    def from_dict(cls, variant_id: str, d: dict) -> "PromptVariant":
        query_mode = d.get("query_mode", "base")
        if query_mode not in QUERY_MODES:
            raise ValueError(
                f"variant {variant_id!r}: query_mode must be one of {QUERY_MODES}, "
                f"got {query_mode!r}"
            )
        return cls(
            variant_id=variant_id,
            instructions=d["instructions"],
            description=d.get("description", ""),
            query_mode=query_mode,
        )


def load_variants(path: str | Path) -> dict[str, PromptVariant]:
    """Load a JSON file mapping variant_id -> {"instructions": ..., "description": ...}.

    Raises ValueError if the file doesn't define DEFAULT_VARIANT_ID ("default"),
    since the eval runner and judge harness both assume a baseline variant exists
    to compare everything else against.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, dict) or not raw:
        raise ValueError(f"{path}: expected a non-empty JSON object of variant_id -> {{...}}")
    variants = {vid: PromptVariant.from_dict(vid, d) for vid, d in raw.items()}
    if DEFAULT_VARIANT_ID not in variants:
        raise ValueError(
            f"{path}: missing required {DEFAULT_VARIANT_ID!r} variant "
            f"(the baseline every other variant is compared against)"
        )
    return variants
