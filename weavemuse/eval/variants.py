"""Loader for prompt variants: named `instructions` strings passed to the
manager `CodeAgent`'s `instructions=` constructor kwarg.

`instructions` (not `description`) is what actually changes an agent's own
reasoning: it gets woven into that agent's system prompt template as
`custom_instructions` (see smolagents.agents.CodeAgent.initialize_system_prompt).
`description` only affects how a *parent* agent's prompt describes this agent
as a callable tool -- irrelevant here since the manager is the top-level agent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_VARIANT_ID = "default"


@dataclass
class PromptVariant:
    variant_id: str
    instructions: str
    description: str = ""

    @classmethod
    def from_dict(cls, variant_id: str, d: dict) -> "PromptVariant":
        return cls(
            variant_id=variant_id,
            instructions=d["instructions"],
            description=d.get("description", ""),
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
