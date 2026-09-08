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

## 9. Batch of fixes after the fabrication smoke (upfront, not one-per-run)

The 14B sub-agent kept emitting a fabricated tonal plan on step 1 instead of
calling a tool. Root cause: smolagents' default managed-agent task wrapper
("your final_answer WILL HAVE to contain ### 1 / ### 2 / ### 3, everything else
is lost") pushes a mid-size model straight to filling that template from
imagination. Verified against the smolagents 1.21.3 source before changing
anything this round:

- `create_musicology_agent` now **overrides** `prompt_templates["managed_agent"]
  ["task"]` with a version that demands a tool call before any answer, in code,
  no invented values. Also adds an `instructions=` guardrail and raises the
  sub-agent `max_steps` 12 -> 16 (paged `get_harmony` walks for SW2).
- Manager `additional_authorized_imports` was `[]` while the new prompts tell it
  to `json.loads` tool output -> added json/math/statistics/collections/re.
- `run_judge.py` no longer hard-requires HF_TOKEN for `--backend local`.
- Judge now takes `--dataset`: each task's `metadata.reference` (when present) is
  shown to the judge as the expert answer to score task_success against.
- `scripts/build_references.py` computes those references for the scriptable
  question types (sw1_modulation, sw2_cadences, sw4_strophic, mw1_period_norm)
  straight from the Didone tools, into `data/eval/tasks_musicology.local.jsonl`
  (gitignored -- derived from the confidential corpus). 37/56 tasks get a
  reference; sw3/sw5/mw2/mw3 stay for human calibration grading.
- `get_tonal_plan` reference note: flags when metadata home key and the
  tonal-plan opening region disagree (e.g. record 0001).

## 10. Intervention moved from the manager prompt into the question (per-template)

The `expert` manager-prompt prefix (section 6) was a weak lever: one fixed
block covering all nine question types, prepended to every task, so on any
single question 8/9 of it was irrelevant. The smoke tests showed no expert
benefit. Reworked the design so the domain intervention is **per question
type, carried in the query**, not in the manager's system prompt.

- **`data/eval/expert_prompts_musicology.json`** (new) -- `{question_template
  -> method block}`. Each block gives the musicological concepts and the
  procedure for that question type, and nothing else -- it does not restate or
  re-scope the question, and names no tool and no sub-agent (choosing which
  capability to use, and whether to look up harmonic/structural data, is the
  agent's call and part of what is scored). Writable by a musicologist with no
  knowledge of the system.
- **`weavemuse/eval/variants.py`** -- `PromptVariant` gained `query_mode`
  (`"base"` | `"expert"`). `"expert"` appends the block for the task's
  `category` to the query verbatim (`runner._compose_query`:
  `query + "\n\n" + block`).
- **`data/eval/variants_musicology.json`** -- `default` and `expert` now carry
  **identical `instructions`** (the operational block only) and differ solely
  in `query_mode`. The manager prompt is no longer a variable, so it cannot
  confound the comparison.
- **`scripts/build_musicology_tasks.py`** -- the old `Q` table split into
  `Q_BASE` (-> dataset `query`) and `Q_EXPERT` (-> expert-prompts file).
  Regenerates both artifacts.
- **`scripts/run_eval.py`** -- `--expert-prompts` flag; auto-detects
  `data/eval/expert_prompts_musicology.json`. `run_sweep` refuses to start if
  a variant is `query_mode="expert"` but no expert-prompts file loaded.
- Traces now record `base_query` (verbatim task query), `query` (what the
  agent received: base + block in expert mode), and `query_mode`.

**Split rule (what goes in base vs expert).** `default` = the naive question a
non-specialist would ask: no concept definitions, no procedure (e.g. "does it
modulate? if so, how many times, and to which keys?"). `expert` = that exact
question + a block giving the musicological concepts and the analytical
procedure a specialist would apply (e.g. "work through the harmonic events,
look for cadences that pivot from one key to another, determine the new key at
each, count the distinct tonal areas"). Designed template by template with a
musicologist; see the per-template entries in `build_musicology_tasks.py`.

**Confounder controls.** Between conditions, the question sentence is
byte-identical; the manager `instructions` are byte-identical; same 56 tasks,
same arias, same backbone. The expert block adds only musicological method --
no operational/formatting guidance (that lives in the shared `instructions`),
no capability names, no restatement of the question.

**Accepted limitation -- grading the four computed-reference templates.** For
sw1/sw2/sw4/mw1 the judge compares against a reference computed under a fixed
criterion (`build_references.py`: e.g. sw1 counts only cadence-confirmed key
changes). The `default` question does not state that criterion -- a naive user
would not -- so a `default` answer that counts under a looser but defensible
reading will diverge from the reference. This is treated as a real effect of
not having expertise, not a harness bug; the judge rubric should credit a
defensible answer rather than exact number-matching. The five open templates
(sw3/sw5/mw2/mw3/mw4) have no computed reference and are unaffected.

**Judge -- not yet updated.** `judge.py` still labels `variant_instructions`
as "what varies across the study"; that line is now stale (the query varies,
not the instructions) and blinding (show the judge `base_query` only) is a
deliberate open decision. Deferred to the judging pass; traces already carry
`base_query` so no re-run is needed for it.

## Open items

- Run the first full 56×2 sweep under the new per-question design (Colab A100).
- Update `judge.py` for the new design: relabel the "what varies" line; decide
  whether the judge sees `base_query` (condition-blind) or the composed query.
- Re-run the smoke (smoke4) and confirm the sub-agent now calls tools first.
- If 14B still fabricates: a pre-quantized 32B (AWQ, ~19GB) or accept it as a
  finding about the harness floor (Colab can't host a bigger un-quantized model).
- Hand-grade ~5 final answers to calibrate the judge against the references.
- `scripts/summarize_eval.py` to pivot judge scores to task × variant × criterion.

---

## For the write-up (the parts that matter)

**Setup.** Backbone: Qwen2.5-Coder-14B, 4-bit, local on a Colab A100. 7B was
below the usable floor (fabricated tool use); 32B did not fit Colab disk -- so
14B is a stated capability ceiling on the results. Manager runs with the
analysis agents only (`musicology_analysis_agent`, `chat_musician`, web search);
generative/audio agents excluded because no task needs them.

**Method.** Same 56 questions under two conditions that differ only in the
question text (the manager prompt is held identical): `default` = the naive
question a non-specialist would ask; `expert` = that same question plus a
per-question-type block giving the musicological concepts and the analytical
procedure a specialist would apply, naming no tool or sub-agent. Designed
template by template with a musicologist. See section 10 for why the
intervention moved from the manager prompt into the question, and the accepted
limitation on grading the computed-reference templates. Traces captured,
scored by an LLM judge (Claude Haiku remote) on a 4-criterion rubric. For
SW1/SW2/SW4/MW1, the judge is given a reference answer computed directly from
the Didone data (`build_references.py`); SW3/SW5/MW2/MW3 are human-graded on a
calibration sample.

**Design choices that affect validity.**
- Tools return evidence (tonal plans, chord tables, section structure), never
  conclusions -- cadence typing, style and norm judgements are the agent's, and
  are the dependent variable.
- "Modulation count" is criterion-dependent; SW1 fixes the criterion
  (cadentially confirmed key area, >= ~8 bars).
- smolagents' default managed-sub-agent prompt template had to be replaced: the
  stock "your answer must contain sections 1/2/3, everything else is lost"
  wording drives a mid-size model to fabricate a structured answer before
  calling any tool. Worth reporting as a scaffolding finding in its own right.
- 12-aria purposive sample, frozen pre-run (small; stated limitation).

**Preliminary observations (smoke tests, n small -- not results).** 14B routes
to the right sub-agent reliably. On SW1 both prompt conditions produced weak
analysis: literal counting of data fields, self-contradiction, and (pre-fix)
fabrication before tool use. No clear expert-prompt benefit on the single task
tested so far.
