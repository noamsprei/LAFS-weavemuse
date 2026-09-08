"""LLM-judge harness: score a saved trace JSON against a rubric.

Never feeds the judge the raw `messages` field verbatim -- each ActionStep's
`model_input_messages` contains the *entire growing conversation up to that
step* (smolagents resends full history every step), so the raw trace is
highly redundant and would blow the judge's context / token budget for no
benefit. `summarize_trace_for_judge()` extracts just the load-bearing fields
per step first.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Protocol

from weavemuse.eval.dataset import EvalTask

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class JudgeBackend(Protocol):
    def score(self, prompt: str) -> str:
        """Send `prompt` to the judge model, return its raw text response."""
        ...


class LocalJudgeBackend:
    """Wraps an already-built weavemuse TransformersModel instance. Zero API
    cost. Caveat, surfaced in output rather than hidden: if the SAME weak
    local model that generated a trace also judges it, expect it to share
    that model's blind spots (e.g. it may not recognize its own hallucinated
    tool names as wrong).
    """

    def __init__(self, model):
        self._model = model

    def score(self, prompt: str) -> str:
        from smolagents import ChatMessage, MessageRole

        # smolagents' get_clean_message_list() expects structured content
        # blocks ([{"type": "text", "text": ...}]), not a bare string -- a
        # plain str here fails with "TypeError: string indices must be
        # integers" deep inside TransformersModel.generate().
        chat_message = ChatMessage(
            role=MessageRole.USER, content=[{"type": "text", "text": prompt}]
        )
        message = self._model.generate([chat_message])
        return message.content or ""


class RemoteJudgeBackend:
    """Wraps anthropic.Anthropic() -- SDK already a repo dependency,
    ANTHROPIC_API_KEY already a recognized field in weavemuse/utils/config.py.
    Defaults to claude-haiku-4-5 ($1/$5 per 1M tokens, the cheapest current
    Claude model) given a stated tight token budget; pass a different
    model_id (e.g. claude-sonnet-5) for higher-fidelity spot checks on a
    small subset.
    """

    def __init__(self, model_id: str = "claude-haiku-4-5", max_tokens: int = 1024):
        import anthropic

        self._client = anthropic.Anthropic()
        self._model_id = model_id
        self._max_tokens = max_tokens

    def score(self, prompt: str) -> str:
        response = self._client.messages.create(
            model=self._model_id,
            max_tokens=self._max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


def summarize_trace_for_judge(trace_record: dict, max_chars_per_step: int = 800) -> str:
    """Walk trace_record["messages"] (ActionStep dicts from
    agent.memory.get_full_steps()) and extract only step_number, code_action,
    tool_calls, observations, error, is_final_answer, action_output per step
    -- explicitly skipping model_input_messages (the redundant growing-context
    field) -- truncating long fields.
    """

    def _truncate(value) -> str:
        s = value if isinstance(value, str) else json.dumps(value, default=str)
        if len(s) > max_chars_per_step:
            return s[:max_chars_per_step] + f"... [truncated, {len(s)} chars total]"
        return s

    lines: list[str] = []
    for step in trace_record.get("messages", []):
        if "step_number" not in step:
            continue  # skip TaskStep/PlanningStep entries, only summarize ActionSteps
        lines.append(f"--- Step {step.get('step_number')} ---")
        if step.get("code_action"):
            lines.append(f"Code: {_truncate(step['code_action'])}")
        if step.get("tool_calls"):
            lines.append(f"Tool calls: {_truncate(step['tool_calls'])}")
        if step.get("observations"):
            lines.append(f"Observations: {_truncate(step['observations'])}")
        if step.get("error"):
            lines.append(f"Error: {_truncate(step['error'])}")
        if step.get("action_output") is not None:
            lines.append(f"Action output: {_truncate(step['action_output'])}")
        if step.get("is_final_answer"):
            lines.append("(this step is the final answer)")
    return "\n".join(lines)


# Per-category technical-definition paragraphs the judge should hold the trace
# to, lifted verbatim from data/eval/variants_musicology.json's "expert" variant
# (the musicological framing given to the agent itself in that condition) so the
# judge and that prompt agree on what "correct" means for each question
# template. Duplicated here rather than imported: variants_musicology.json is a
# data file describing an *agent* prompt variant, not a code module, and this is
# the only other place that needs these paragraphs. Keep the two in sync by hand
# if either is edited.
_CATEGORY_PARAGRAPHS = {
    "modulation": (
        "Modulation and tonal scheme. An aria modulates when the music leaves "
        "the home key and confirms another key with a cadence; a brief "
        "tonicisation in passing does not count. To assess: establish the home "
        "key; go through the tonal design section by section; for each section "
        "look at the harmonic function it closes on -- that closing function is "
        "the cadential evidence for whether a new key was confirmed there; a "
        "very short section with no cadential close in its own key is a "
        "tonicisation, not a modulation; count only the confirmed key changes; "
        "report the ordered sequence of key areas. A da capo aria restates its "
        "opening key motion late in the plan; that restatement is the return "
        "home, not a new modulation. 'No modulation beyond the dominant' is a "
        "legitimate finding."
    ),
    "cadences": (
        "Cadences. A cadence is the harmonic-melodic close that articulates the "
        "end of a phrase. Find them by locating points of arrival or repose and "
        "reading the progression that leads into each. Classify each by that "
        "progression: perfect (dominant to tonic), imperfect (a phrase ending "
        "on the dominant), interrupted (dominant to submediant), plagal "
        "(subdominant to tonic)."
    ),
    "typicality": (
        "Typicality and period norms. A feature is 'typical of a period' only "
        "in relation to a body of contemporaneous works. To judge whether an "
        "aria's tonal scheme, opening progression, or formal plan is typical, "
        "compare it against a representative sample of other arias from the "
        "same time -- or by the same composer, or setting the same text -- and "
        "compare the distribution, not a single case. One aria cannot "
        "establish a norm."
    ),
    "galant_baroque": (
        "Galant versus Baroque character. These are bundles of tendencies, not "
        "single traits. Baroque writing tends toward continuous spinning-out of "
        "a line, denser counterpoint, motoric and consistent rhythm, and few "
        "full stops. Galant writing tends toward periodic phrasing with "
        "frequent clear cadences, lighter melody-and-accompaniment texture, and "
        "more sectional articulation. Deciding which pole an aria leans toward "
        "requires weighing several concrete, checkable features actually "
        "retrieved for this aria -- not the composer's general reputation or "
        "the year alone."
    ),
    "stormy_pastoral": (
        "Stormy versus pastoral affect. Stormy (Sturm und Drang) writing tends "
        "toward fast tempo, minor mode, dense and agitated note motion, and "
        "driving rhythm. Pastoral writing tends toward moderate or slow tempo, "
        "lilting compound or triple metre, transparent scoring, and gentle "
        "motion. Weighing the concrete retrieved features is required; "
        "deciding from the composer or the year alone is not sufficient."
    ),
    "strophic": (
        "Strophic form from the music alone. The stanzas of the text usually "
        "leave a musical trace. A return to the opening material in the home "
        "key marks a fresh statement of the main strophe (an A section); a "
        "contrasting passage that begins and ends away from the tonic is "
        "typically the middle strophe (a B section). Look for an exact "
        "restatement of the opening tonal motion to locate the repeats."
    ),
}

# Which paragraph(s) apply to each question_template/category. mw3 gets both
# typicality (it's a comparison-against-a-sample question) and strophic (the
# thing being compared is the strophic/formal plan).
_CATEGORY_TO_PARAGRAPHS = {
    "sw1_modulation": ["modulation"],
    "sw2_cadences": ["cadences"],
    "sw3_style": ["galant_baroque"],
    "sw4_strophic": ["strophic"],
    "sw5_affect": ["stormy_pastoral"],
    "mw1_period_norm": ["typicality"],
    "mw2_bsection_opening": ["typicality"],
    "mw3_composer_textplan": ["typicality", "strophic"],
    "mw4_year_cohort": ["typicality"],
}


def _category_definitions_block(category: str | None) -> str:
    keys = _CATEGORY_TO_PARAGRAPHS.get(category or "", [])
    if not keys:
        return ""
    paragraphs = "\n\n".join(_CATEGORY_PARAGRAPHS[k] for k in keys)
    return (
        f"\nThe technical definition(s) that apply to this question type "
        f"({category}) -- hold the trace to these, not to a looser everyday "
        f"reading of the terms:\n{paragraphs}\n"
    )


def build_judge_prompt(
    rubric: dict, task: EvalTask, trace_record: dict, reference: str | None = None
) -> str:
    by_group: dict[str, list[str]] = {}
    for name, c in rubric["criteria"].items():
        by_group.setdefault(c.get("group", "other"), []).append(
            f"- {name}: {c['description']} (scale: {c['scale']})"
        )
    _GROUP_TITLES = {
        "agentic_flow": "Agentic-flow criteria (tool use, process, efficiency)",
        "musicology": "Musicology criteria (domain-specific correctness)",
        "other": "Other criteria",
    }
    criteria_desc = "\n\n".join(
        f"{_GROUP_TITLES.get(group, group)}:\n" + "\n".join(lines)
        for group, lines in by_group.items()
    )
    trace_summary = summarize_trace_for_judge(trace_record)
    reference_block = (
        f"\nReference (a domain expert's answer / analytical plan for this task -- "
        f"judge task_success and reasoning against this, not against your own "
        f"guess):\n{reference}\n"
        if reference else ""
    )
    definitions_block = _category_definitions_block(task.category)
    example_json = "{\n" + ",\n".join(
        f'  "{name}": {{"score": <int 0-5>, "rationale": "<one sentence>"}}'
        for name in rubric["criteria"]
    ) + ',\n  "notes": "<any other observations>"\n}'
    return f"""You are evaluating the quality of an AI agent's execution trace on a task.

Task query: {task.query}
{reference_block}{definitions_block}
Prompt/instructions the agent was given (this is what varies across the study -- \
judge how well the agent's behavior reflects these instructions, not just whether \
it succeeded):
{trace_record['variant_instructions']}

Agent's final output: {trace_record['output']}
Final state: {trace_record['state']}

Execution trace (step-by-step reasoning, tool calls, observations):
{trace_summary}

Score the trace on each of these criteria:
{criteria_desc}

Respond with ONLY a JSON object (no other text) in exactly this shape:
{example_json}"""


def parse_judge_json(raw_text: str) -> dict:
    """Tolerant parse: try a fenced ```json block first, else raw json.loads,
    else return a {parse_error, raw} dict rather than raising -- a malformed
    judge response should never crash the whole batch.
    """
    candidates = [raw_text.strip()]
    fence_match = _JSON_FENCE_RE.search(raw_text)
    if fence_match:
        candidates.insert(0, fence_match.group(1).strip())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return {"parse_error": "could not parse judge response as JSON", "raw": raw_text}


def score_trace_file(
    path: str | Path,
    backend: JudgeBackend,
    rubric: dict,
    judge_backend_name: str,
    judge_model_id: str,
    tasks_by_id: dict | None = None,
) -> dict:
    """Load one trace JSON, build the judge prompt, score it, return the
    score record (same shape written to disk by scripts/run_judge.py).

    If `tasks_by_id` (task_id -> task dict from the dataset) is given and the
    task's metadata carries a "reference" field, that reference is shown to
    the judge as the expert answer to score task_success against.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        trace_record = json.load(f)

    task_dict = (tasks_by_id or {}).get(trace_record["task_id"]) or {}
    task = EvalTask(
        task_id=trace_record["task_id"],
        query=trace_record["query"],
        category=task_dict.get("category"),
    )
    reference = (task_dict.get("metadata") or {}).get("reference")
    prompt = build_judge_prompt(rubric, task, trace_record, reference=reference)
    raw_response = backend.score(prompt)
    parsed = parse_judge_json(raw_response)

    scores = {k: v for k, v in parsed.items() if k in rubric["criteria"]}
    # Attach objective signals (step_count, total_tokens) alongside the
    # judge's subjective efficiency score, per the rubric's own guidance.
    if "efficiency" in scores:
        step_count = sum(1 for m in trace_record.get("messages", []) if "step_number" in m)
        scores["efficiency"]["step_count"] = step_count
        token_usage = trace_record.get("token_usage") or {}
        scores["efficiency"]["total_tokens"] = token_usage.get("total_tokens")

    return {
        "task_id": trace_record["task_id"],
        "variant_id": trace_record["variant_id"],
        "judge_backend": judge_backend_name,
        "judge_model_id": judge_model_id,
        "rubric_version": rubric.get("version", "unknown"),
        "scores": scores,
        "notes": parsed.get("notes"),
        "parse_error": parsed.get("parse_error"),
        "raw_judge_response": raw_response,
    }
