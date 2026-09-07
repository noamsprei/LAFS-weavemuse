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
terminal entrypoints are unaffected. The eval runner picks it up through its
existing `get_weavemuse_agents_and_tools()` call.

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

## Open items

- Extend the LLM judge to score against per-question expectations (the `metadata`
  in each task is preserved but not yet read by `judge.py`).
- Run the sweep on the GPU box; hand-grade ~5 final answers to calibrate the judge.
- `scripts/summarize_eval.py` to pivot judge scores to task × variant × criterion.
