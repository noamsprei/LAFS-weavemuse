# Musicology prompt-intervention study — project log

Short chronological log of what was built and why. Newest section last.

## Research question

How do domain-specific prompt interventions affect the decomposition, tool
routing, and reasoning quality of the WeaveMuse manager agent on
musicological-analysis tasks? Design: same questions, two prompt conditions
(baseline vs expert), compare the execution traces.

## 1. Corpus choice — Didone dataset

Picked the Didone corpus (~2,500 18th-c. Italian opera arias) because it ships
pre-computed structured analyses (tonal plans, chord-by-chord Roman-numeral
harmony, section/text structure) and because multiple composers set the *same*
libretto texts across decades — a built-in control for the "is this typical of
its period" questions. The MIDI files are not used at query time; every tool is
a CSV lookup. Corpus data is confidential and gitignored.

## 2. Test sample — 12 arias, frozen

`data/eval/sample_12.jsonl`. Purposive, not random: 3 same-text clusters × 3
settings (1724–1770) for the comparison questions, plus 3 minor-mode arias for
the style/affect questions (the corpus is 98.7% major), across 7 composers with
repeats for the composer-similarity question. Frozen before any agent run to
avoid selection bias; small purposive sample is a stated limitation. 12 rather
than 20 to keep the hand-graded calibration set tractable.

## 3. Query tools — `weavemuse/tools/didone_tools.py`

Six single-aria tools + three batch tools. Key decision: tools return
**evidence, not conclusions**. There is no `get_cadences` or `get_style` tool —
the agent must infer cadences from `get_harmony` and style from raw features.
That inference is the dependent variable. Batch tools (`get_tonal_plans` etc.)
were added because the multi-work questions need one comparison feature for
~40 arias, which would blow `max_steps` one call at a time.

`get_harmony` is capped at 40 measures/call to bound the context growth that
causes OOM on small local backbones.

## 4. Agent wiring

`create_musicology_agent()` in `agents_as_tools.py` builds a `CodeAgent` with
the nine tools; added to the manager's `managed_agents` by default. Degrades to
"absent, with a warning" when the corpus isn't found, so the shipped GUI/
terminal entrypoints are unaffected.

`get_weavemuse_agents_and_tools()` gained an `exclude_agents` parameter (and
`run_eval.py` a `--exclude-agents` flag). For this study the manager keeps
`musicology_analysis_agent`, `chat_musician`, and `web_search_agent`; the
generative and audio agents are excluded — the tasks are symbolic/metadata
analysis with no generation or audio input, so those agents would only ever be
mis-routes. Keeping `chat_musician` (a local model) means the sweep needs a GPU
and `--tool-mode hybrid`. `HF_TOKEN` is now a warning, not a hard requirement,
since no HF-hosted agent is in the set.

## 5. Questions — `data/eval/tasks_musicology.jsonl` (56)

9 templates over the 12 arias. Tiered: *lookup* (SW1 modulation) tests
decomposition/routing; *inference* (SW2 cadences, SW3/SW5 style) tests
reasoning; *retrieval+inference* (MW1–MW4) tests whether the agent recognises a
question needs corpus aggregation. SW1/SW2/MW1 tagged `core` for a first pass.
Terminology fixed to British cadence terms (perfect/imperfect/interrupted/
plagal) so no soprano-voice data is needed. SW3 dropped texture and mode as
cues (not measurable / too rare) and is left open-ended on which features
matter.

## 6. Prompts — `data/eval/variants_musicology.json`

`default` = stock manager prompt (control). `expert` = same operational text +
a musicological block: concept definitions and the analytical decomposition
each question type needs. Deliberately names **no tools and no sub-agents** —
mapping a sub-goal like "compare against contemporaneous arias" onto an actual
capability is the agent's job and part of what is scored. Test applied: a
musicologist with no knowledge of the codebase could have written it.

## 7. Colab smoke test (1 task, both variants)

Ran `sw1_modulation__0012` on an A100 with the 14B backbone. Pipeline worked end
to end: correct routing to `musicology_analysis_agent`, both variants finished.
But the analysis quality was poor in both conditions, for a fixable reason: the
tools returned CSV-ish text blobs and the agent kept trying to `str.split(',')`
them, mis-indexed, and once even reported the CSV header row ("region_label") as
a key area.

Fixes applied:
- **All Didone tools now return JSON**, not text tables. Tool descriptions say
  "read the JSON directly; do not string-split". `get_tonal_plan` surfaces
  `home_key` and a per-segment `closes_on` (the cadential evidence).
- **Both prompt variants** gained a shared operational line: results are JSON,
  parse with `json.loads`, don't string-split, don't redo a sub-agent's work.
  (Operational, not musicological -- kept identical in both so it doesn't
  confound the intervention.)
- **Expert prompt** modulation paragraph now points at the section-by-section
  closing function as cadence evidence and notes the da capo restatement is a
  return, not a new modulation -- still names no tools.

Known, not yet addressed: the managed-agent "### 1/2/3" response template
induces some hallucinated padding in section 3. It is constant across both
conditions so it does not confound the comparison; left alone for now.

## 8. Second smoke (JSON tools) + tonal-plan fixes

JSON tools fixed the string-parsing crashes, but both variants still answered
`sw1_modulation__0012` badly and in opposite directions: default computed "0
modulations" (broken loop), expert computed "7" (counted every segment whose
annotation `modulation_type != "none"`, incl. the opening) and printed
"modulates to None" six times. Root causes:

- `get_tonal_plan` returned `absolute_key` only for segment 1 (blank for the
  rest in the source), so 6/7 segments showed key `null`.
- The `modulation_type` field ("initial" / "explicit_region_marker") is an
  annotation-boundary label, not a modulation-vs-tonicisation judgement, and
  the agent counted it literally.
- "how many times does it modulate" has no single answer -- strict (cadentially
  confirmed) gives ~1-2, liberal (every region visited) gives ~6-7.

Fixes:
- `get_tonal_plan` now computes each segment's absolute key from the home key +
  the Roman-numeral region label (via music21), adds `length_measures`, renames
  `modulation_type` -> `boundary_type`, drops `parent_region`. Every segment now
  has a real key and a length; the tool still does not label modulation vs
  tonicisation (that is the agent's call).
- SW1 rewritten to define the target: count a key area as a confirmed
  modulation only if a phrase cadences (perfect or half) in that key; short
  passed-through regions do not count; the da capo return is not a modulation.
  Now gradeable.

Also observed: 7B backbone is below the usable floor (hallucinated `import
requests` to fetch a fake corpus URL). 14B on A100 routes correctly; 32B may be
worth trying for cleaner tool-use code.

## Open items

- Re-run the smoke task with the JSON tools + updated prompts; confirm cleaner
  traces before the core sweep.
- Extend the LLM judge to score against per-question expectations (the `metadata`
  in each task is preserved but not yet read by `judge.py`).
- Hand-grade ~5 final answers to calibrate the judge.
- `scripts/summarize_eval.py` to pivot judge scores to task × variant × criterion.
