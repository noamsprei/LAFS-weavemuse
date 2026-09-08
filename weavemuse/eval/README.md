# WeaveMuse trace-quality evaluation harness

Runs WeaveMuse's manager `CodeAgent` over a dataset of tasks under different
prompt ("instructions") variants, captures the full step-by-step execution
trace of each run, and scores those traces with an LLM judge. Built to
compare the framework's **default** manager prompt against **modified**
prompts and see whether/how trace quality changes.

Fully additive: nothing here is imported by, or modifies, the rest of
WeaveMuse (`app.py`, `weavemuse/interfaces/*`, `scripts/quickstart_*.py`,
`weavemuse/agents/agents_as_tools.py` are all untouched).

## How it works

Two separate CLI scripts, joined only by JSON files on disk:

1. **`scripts/run_eval.py`** — loads the local backbone LLM once, then loops
   every `(task, variant)` pair: builds a *fresh* manager agent for that
   pair, runs it, saves the full trace as JSON, releases GPU memory, moves on.
2. **`scripts/run_judge.py`** — a separate process, run afterward. Reads the
   saved trace JSONs and scores each one against a rubric using an LLM judge
   (local or remote backend).

Running them as separate processes is deliberate: phase 1's GPU memory is
fully released when that process exits, before phase 2 (which may also load
a model, if `--backend local`) allocates anything.

## Quick start

```bash
# 1. Make sure HF_TOKEN is set in .env (needed for the default remote tools)
# 2. Generate traces for the 3-task smoke dataset x 2 variants (6 runs):
.venv/bin/python scripts/run_eval.py \
  --dataset data/eval/tasks_smoke.jsonl \
  --variants data/eval/variants_smoke.json \
  --run-id smoke_test

# 3. Score them (starts local, zero API cost):
.venv/bin/python scripts/run_judge.py \
  --traces-dir outputs/eval/smoke_test/traces \
  --backend local --limit 2

# 4. Once you trust the pipeline, try 2 remote-judged files (~$0.01-0.05):
.venv/bin/python scripts/run_judge.py \
  --traces-dir outputs/eval/smoke_test/traces \
  --backend remote --limit 2

# 5. Compare a task across variants directly:
diff outputs/eval/smoke_test/traces/t001/default.json \
     outputs/eval/smoke_test/traces/t001/concise_reasoning.json
```

Output layout:
```
outputs/eval/<run_id>/
  manifest.json                          # run config + per-pair summary (state, tokens, VRAM)
  traces/<task_id>/<variant_id>.json     # full trace: RunResult (output/state/messages/tokens/timing)
  scores/<task_id>/<variant_id>.judge.json   # judge's scores against the rubric
```
`outputs/` is gitignored — these are run artifacts, not checked in.

Each trace also carries two fields beyond the manager's own flattened
`messages`, from `weavemuse/eval/tool_tracing.py` (always on, no flag
needed):
- **`tool_calls`** — an ordered ledger of every individual tool call *and*
  every manager→sub-agent call (name, args, action `called`/`error`,
  truncated output, timing) — this is the only place to see e.g. the exact
  `audio_flamingo` invocation; the manager's own `messages` field only ever
  records a single synthetic `python_interpreter` step per turn (the whole
  generated code blob), never the individual calls inside it.
- **`sub_agent_traces`** — each managed sub-agent's own
  `memory.get_full_steps()` (e.g. `audio_analysis_agent`'s own
  Thought/Code/Observation cycle, including which tool it tried first and
  what it fell back to) — previously invisible; only the sub-agent's bare
  final-answer text ever reached the manager's trace before this. **Caveat**:
  if a sub-agent is invoked more than once within one task, only its *last*
  invocation's steps survive here (smolagents resets a sub-agent's memory
  on each call) — acceptable given these sub-agents cap at `max_steps=1` or
  `2` and are typically invoked once per task.

## What to modify for your actual study

- **`data/eval/tasks_smoke.jsonl`** → write your own dataset here (or point
  `--dataset` at a new file). One JSON object per line:
  ```json
  {"task_id": "t004", "query": "...", "category": "...", "tags": [], "metadata": {}, "attachments": {}}
  ```
  `task_id` must be unique within the file (the loader raises on duplicates).

  **`attachments`** (optional) is how to give a task local files (audio,
  MIDI, images, ...) that the agent should actually use, e.g.
  `{"audio_file": "audio/foo.wav"}` -- paths are resolved relative to the
  dataset file's own directory and validated to exist at load time
  (`load_tasks()` raises a clear error naming the task/key/path otherwise).
  `run_one()` passes `attachments` to `agent.run(additional_args=...)`, so
  each key becomes both literal text in the task prompt and a real Python
  variable the manager's generated code can reference.

  **Name each key after the actual tool parameter it will end up at**, e.g.
  `audio_file` (matching `AudioFlamingoTool.forward(audio_file=...)` /
  `AudioAnalysisTool.forward(audio_file=...)`), not a generic name like
  `audio`. This was confirmed the hard way: `additional_args` values keep
  their dict key as the variable name at every hop (manager's own scope →
  the sub-agent's own scope via its own `additional_args={...}` call →
  whatever keyword the LLM then uses on the actual tool call) -- an LLM
  asked to "forward the exact path" will frequently forward it under its
  existing variable name rather than renaming it to the tool's real
  parameter, producing a `TypeError: ...forward() got an unexpected keyword
  argument 'audio'` a couple of steps in. Naming the attachment key after
  the tool's parameter from the start makes every hop a same-name passthrough
  and sidesteps this failure mode entirely, rather than relying on prompt
  wording to get an LLM to rename a variable correctly.

  **Never put an answer/reference field (e.g. `ground_truth`) in
  `attachments`** -- anything there reaches the model. Keep those in
  `metadata` instead, which stays agent-invisible bookkeeping (used only by
  you / a future judge extension, never passed to `agent.run()`).

- **`data/eval/variants_smoke.json`** → this is where your "default vs.
  modified prompt" comparison actually lives. Every variant file MUST define
  a `"default"` entry (the baseline). Add your modified prompt(s) as
  additional keys:
  ```json
  {
    "default": {"instructions": "...", "description": "..."},
    "your_variant_name": {"instructions": "...", "description": "..."}
  }
  ```
  **Important**: `instructions` is what you're actually varying — it's woven
  directly into the manager agent's own system prompt (see
  `weavemuse/eval/variants.py`'s module docstring for why this, and not
  `description`, is the correct lever). If your study is instead about
  varying `description` for a specific *sub-agent* (not the top-level
  manager), that's a different, deeper change not covered by this harness as
  built — ask before assuming it's a trivial extension.

- **`weavemuse/eval/rubrics/default.json`** → the judge's scoring criteria.
  Edit `criteria` freely (add/remove/reword) — `judge.py`'s
  `build_judge_prompt()` builds both the criteria list and the example
  response JSON from whatever's in the file you pass via `--rubric`, nothing
  is hardcoded to specific criterion names. Each criterion also carries a
  `"group"` tag (`"agentic_flow"` — tool use, decomposition, efficiency — or
  `"musicology"` — domain-specific correctness for the Didone-corpus study);
  the judge prompt renders the two groups under separate headers, and
  `scripts/run_judge.py`'s stdout summary is grouped the same way. This is a
  single rubric/single judge call either way — the grouping is for readability
  and later slicing of `scores/**/*.judge.json`, not two separate passes.

- **Category-specific technical definitions**: for the musicology study,
  `judge.py`'s `_CATEGORY_TO_PARAGRAPHS` maps each task's `category` (e.g.
  `sw1_modulation`, `sw3_style`) to the relevant technical-definition
  paragraph(s) (the modulation/cadence rule, the galant-vs-Baroque feature
  bundle, etc. — the same text used in `data/eval/variants_musicology.json`'s
  `"expert"` prompt variant), and injects the matching paragraph into the
  judge prompt so the judge checks the trace against the actual rule for that
  question type instead of guessing. This requires `score_trace_file()` to be
  given `tasks_by_id` (i.e. run with `--dataset`) — without it, `task.category`
  is `None` and no definition paragraph is injected.

- **`weavemuse/eval/judge.py`'s `build_judge_prompt()`** → if you want the
  judge to weigh things differently (e.g. ask for a single overall score
  instead of per-criterion), this is the one function to change.

## What to pay attention to

- **VRAM is genuinely tight on an 8GB card, and OOM is a real, observed
  outcome, not just a theoretical risk.** `smolagents`' `CodeAgent` resends
  the *entire* growing conversation every step (no cross-step KV-cache
  reuse) — this was observed to grow input tokens from 2,725 (step 1) to
  19,721 (step 5) within a single task. `run_eval.py` mitigates this by
  defaulting `--tool-mode remote` (avoids the *large* local music models —
  StableAudio ~5GB, ChatMusician ~4-8GB) and calling `release_gpu_memory()`
  after every `(task, variant)` pair — verified in practice: free VRAM
  returned to the identical value after every pair in a 6-pair sweep, so
  there's no *run-over-run* creep. But **`RemoteNotaGenTool` still loads a
  small local model despite its name** (confirmed: ~1GB), and *within* a
  single task, a CUDA OOM did occur during development on this exact 8GB
  card. `run_one()` now catches this (any exception from `agent.run()`) and
  records `state="error"` with the exception text in the trace JSON instead
  of crashing the whole sweep — so a single bad pair costs you that one
  result, not the rest of the dataset. Watch `manifest.json`'s
  `free_vram_gb_after` and `error` fields per run if you see this happening
  often; `weavemuse/eval/gpu_guard.py::check_vram_headroom()` still refuses
  to *start* a sweep if there isn't enough free VRAM up front (and names the
  PID holding it, if any — a stale/suspended process holding VRAM
  indefinitely was the root cause of one such pre-sweep incident during
  development, distinct from the within-task OOM above).
- **A leftover process can silently eat your VRAM.** If a prior
  `quickstart_local.py`/`run_eval.py` run opened a blocking window or is
  still alive in another terminal, it keeps its model loaded in VRAM. Check
  with `nvidia-smi --query-compute-apps=pid,process_name,used_memory
  --format=csv` before a new run if one unexpectedly OOMs.
- **Remote-judge token cost.** `--backend remote` calls the Anthropic API and
  costs real money (default `claude-haiku-4-5`, ~$1/$5 per 1M tokens — the
  cheapest current Claude model). Always run with `--limit` first. Estimate
  cost as roughly (number of trace files) × (prompt tokens, usually a few
  thousand once summarized) — `summarize_trace_for_judge()` already strips
  the redundant `model_input_messages` field before it ever reaches the
  judge, but it's still worth spot-checking `raw_judge_response` length in
  one output file before scaling up.
- **Self-judging bias.** `--backend local` reuses the same weak local model
  class that generated the traces. If it's literally the same model, expect
  it to share that model's blind spots (e.g. not recognizing its own
  hallucinated tool name as wrong). Treat local-judge scores as a pipeline
  smoke test, not a trustworthy final result — use `--backend remote` (even
  just a small `--limit` sample) for anything you'll actually report on.
  This is exactly the local-weak-backbone tradeoff the study's premise
  accepts on the generation side too: the weaker model should show *more*
  visible prompt-sensitivity, which is the point — but it also means noisier,
  more repetitive/looping traces (as observed: a 7B model hallucinating a
  nonexistent tool name and hitting `max_steps`) that a judge has to sort
  through.
- **`WebSearchTool` is always live**, even under `--tool-mode remote` (it's
  a plain DuckDuckGo HTTP scrape, no local weights either way). Any task
  that triggers `web_search_agent` introduces a network/non-determinism
  source into an otherwise-controlled comparison. Avoid web-search-shaped
  tasks in your dataset if strict reproducibility matters more than coverage.
- **A fresh manager agent is built for every `(task, variant)` pair** —
  intentional, not a missed optimization. `CodeAgent.run(reset=True)`
  (smolagents' default) only clears step memory, not the Python-executor's
  variable namespace, so reusing one instance across tasks would leak
  variables between them. Rebuilding is cheap because WeaveMuse's tools are
  lazy-loaded — only the shared backbone model (loaded once) is expensive.

## Why not Langfuse?

Investigated and deliberately skipped for this pass. Langfuse (open-source
LLM observability/eval platform) has a native `SmolagentsInstrumentor` that
auto-captures full traces via OpenTelemetry, plus built-in Datasets/
Experiments and LLM-as-a-judge with a comparison UI — functionally
overlapping most of what this harness builds by hand. It needs either
self-hosted infra (Docker Compose: Postgres+ClickHouse+Redis+MinIO) or a
cloud account (your prompts/traces leave the machine), which is more than
this project's stated goals (additive, local-first, tight budget) need. The
plain-JSON design here already captures the same underlying data
(`agent.memory.get_full_steps()`). To add it later without touching anything
else, gate `SmolagentsInstrumentor().instrument()` behind an env var (e.g.
`if os.getenv("LANGFUSE_PUBLIC_KEY"): ...`) near the top of
`scripts/run_eval.py`'s `main()`.

## Extending later (not built, noted for when you need it)

- `pandas` is already a repo dependency — a `scripts/summarize_eval.py` that
  pivots `scores/**/*.judge.json` into a `task_id × variant_id × criterion`
  comparison table would be cheap to add once you have real data to look at.
- The judge rubric/prompt only asks for per-criterion 0-5 scores today. If
  you want pairwise comparison ("which variant's trace is better for this
  task") instead of independent scoring, that's a different
  `build_judge_prompt()` shape, not covered here.
