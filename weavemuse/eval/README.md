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

## What to modify for your actual study

- **`data/eval/tasks_smoke.jsonl`** → write your own dataset here (or point
  `--dataset` at a new file). One JSON object per line:
  ```json
  {"task_id": "t004", "query": "...", "category": "...", "tags": [], "metadata": {}}
  ```
  `task_id` must be unique within the file (the loader raises on duplicates).

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
  `score_trace_file()` reads whatever criteria are in the file you pass via
  `--rubric`, nothing is hardcoded to the four starting criteria.

- **`weavemuse/eval/judge.py`'s `build_judge_prompt()`** → if you want the
  judge to weigh things differently (e.g. ask for a single overall score
  instead of per-criterion), this is the one function to change.

## What to pay attention to

- **VRAM is genuinely tight on an 8GB card.** `smolagents`' `CodeAgent`
  resends the *entire* growing conversation every step (no cross-step
  KV-cache reuse) — this was observed to grow input tokens from 2,725 (step
  1) to 19,721 (step 5) within a single task, pushing VRAM from ~5.5GB to
  ~7.7GB/8.2GB. `run_eval.py` mitigates this by defaulting `--tool-mode
  remote` (frees all VRAM for the backbone; local music tools aren't needed
  for trace research and would compound the pressure) and calling
  `release_gpu_memory()` after every `(task, variant)` pair — but a single
  long/looping task can still get close to the ceiling. Watch
  `manifest.json`'s `free_vram_gb_after` per run if you see slowdowns or
  OOMs; `weavemuse/eval/gpu_guard.py::check_vram_headroom()` will refuse to
  start a sweep if there isn't enough free VRAM to begin with (and will name
  the PID holding it, if any — a stale/suspended process holding VRAM
  indefinitely was the root cause of one such incident during development).
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
