"""Generate the musicology study's dataset + expert-prompt file from the
frozen 12-aria sample.

Writes two artifacts, deterministically (edit the tables and rerun):

  data/eval/tasks_musicology.jsonl        -- one EvalTask per line, `query`
                                             built from Q_BASE (the naive
                                             question every condition gets)
  data/eval/expert_prompts_musicology.json -- {question_template -> method block}
                                             from Q_EXPERT

The study's two conditions:
  default -> the task's base query, verbatim.
  expert  -> base query + the Q_EXPERT block for that question template
             (appended by weavemuse/eval/runner.py at run time, keyed on the
             task's `category`).

Split rule (see weavemuse/eval/STUDY_LOG.md):
  - Q_BASE is the naive question a non-specialist would ask -- no concept
    definitions, no procedure.
  - Q_EXPERT adds ONLY the analytical method a musicologist would apply to that
    question type: the relevant concepts and the procedure. It names no tool
    and no sub-agent -- choosing which capability to use (and whether to look
    up harmonic/structural data) is the agent's call and part of what is
    scored.

No gold answers are embedded here -- correctness is judged post hoc (LLM judge
+ a human calibration sample); computed references live in
data/eval/tasks_musicology.local.jsonl via scripts/build_references.py.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "data" / "eval" / "sample_12.jsonl"
OUT = REPO / "data" / "eval" / "tasks_musicology.jsonl"
EXPERT_OUT = REPO / "data" / "eval" / "expert_prompts_musicology.json"

# base questions ----------------------------------------------------------
# {ref} is filled with "record_id N (Composer's YEAR setting of 'ARIA')".
# Q_BASE is the naive question a non-specialist would ask. The expert condition
# gets exactly this text plus the matching Q_EXPERT block (the analytical method
# a musicologist would apply), appended at run time by runner._compose_query.
Q_BASE = {
    "sw1_modulation": (
        "For the Didone-corpus aria {ref}: does it modulate? If so, how many "
        "times, and to which keys?"
    ),
    "sw2_cadences": (
        "For the Didone-corpus aria {ref}: how many times does it come to a "
        "cadence, and are those cadences mostly alike or varied in kind?"
    ),
    "sw3_style": (
        "For the Didone-corpus aria {ref}: would you describe it as more galant "
        "or more Baroque in style?"
    ),
    "sw4_strophic": (
        "For the Didone-corpus aria {ref}: can you work out its strophic form -- "
        "the pattern of repeated and contrasting sections -- from the music "
        "alone, without the sung text?"
    ),
    "sw5_affect": (
        "For the Didone-corpus aria {ref}: does it come across as stormy and "
        "agitated, or calm and pastoral?"
    ),
    "mw1_period_norm": (
        "For the Didone-corpus aria {ref}: is its handling of keys typical of "
        "arias from the same period, or unusual?"
    ),
    "mw2_bsection_opening": (
        "For the Didone-corpus aria {ref}: does the music that opens its "
        "contrasting middle section start with the same harmony as other "
        "settings of the same text, or does it do something different?"
    ),
    "mw3_composer_textplan": (
        "For the Didone-corpus aria {ref}: is the way it's laid out into "
        "sections similar to how the same composer shapes other arias?"
    ),
    "mw4_year_cohort": (
        "For the Didone-corpus aria {ref}: do most arias composed in the same "
        "year as this one share similar musical characteristics?"
    ),
}

# expert decomposition blocks ------------------------------------------------
# Appended to the base query in the `expert` condition only. The analytical
# method a musicologist would apply to that question type -- concepts and
# procedure, no tool or sub-agent named (choosing which capability to use is
# the agent's call and part of what is scored).
Q_EXPERT = {
    "sw1_modulation": (
        "Approach this as a harmonic analysis: work through the aria's harmonic "
        "events and look for cadences that pivot from one key to another; at each "
        "such point, determine the new key. Then count the distinct tonal areas "
        "established this way and answer, listing the keys in order."
    ),
    "sw2_cadences": (
        "A cadence is the harmonic formula that closes a phrase. Work through "
        "the aria phrase by phrase: at each point of repose, read the "
        "progression leading into it and classify the cadence -- perfect (V-I), "
        "imperfect (phrase ending on V), interrupted (V-vi), or plagal (IV-I). "
        "Then tally the types and see whether one dominates."
    ),
    "sw3_style": (
        "These are bundles of tendencies, not single traits. Baroque style "
        "leans toward counterpoint, continuous music without strong sectional "
        "breaks, and often triple metre and minor mode; galant style leans "
        "toward short balanced phrases with frequent clear cadences, a light "
        "melody-over-accompaniment texture, and clear sectional articulation. "
        "Identify which of these features you can actually observe in this aria, "
        "examine them, and weigh them to place it on the spectrum -- it need not "
        "be purely one or the other."
    ),
    "sw4_strophic": (
        "The text's stanzas usually leave a musical trace: a return to the "
        "opening material in the home key marks a fresh statement of the main "
        "strophe (an A section); a passage that begins and stays away from the "
        "tonic, sitting between two such returns, is the contrasting middle "
        "strophe (a B section). Find the exact restatements of the opening tonal "
        "motion to locate the section boundaries, then read off the A/B sequence "
        "and explain the reasoning."
    ),
    "sw5_affect": (
        "Stormy (Sturm und Drang) writing tends toward fast tempo, minor mode, "
        "agitated and dense motion, and driving rhythm; pastoral writing tends "
        "toward a moderate or slow tempo, a lilting compound or triple metre, "
        "light transparent texture, and gentle motion. Check which of these "
        "features the aria actually shows, examine them, and weigh them rather "
        "than judging on any single one."
    ),
    "mw1_period_norm": (
        "\"Typical\" only means something relative to a body of contemporaneous "
        "works, so judge it from a distribution, not from this aria alone. "
        "Characterise this aria's tonal scheme -- its home key, the sequence of "
        "key areas it visits, and whether those stay within closely related "
        "keys. Build the same characterisation for a representative sample of "
        "other arias from around the same time, and compare: is this aria's "
        "scheme common in that sample, at its edge, or an outlier?"
    ),
    "mw2_bsection_opening": (
        "The B section is the contrasting middle of the aria, framed away from "
        "the home key. Identify where it begins and read the chord progression "
        "it opens with. Get the corresponding opening progression from the other "
        "settings of the same text, and compare -- is this opening gesture "
        "shared across the settings or particular to this one?"
    ),
    "mw3_composer_textplan": (
        "Trace the section boundaries from the music: mark where the opening "
        "material returns in the home key, and where a passage sits away from "
        "the tonic between two such returns, and note any da capo return. Do the "
        "same reading for a sample of the same composer's other arias, then "
        "compare the resulting layouts -- is this one characteristic of the "
        "composer, or an outlier?"
    ),
    "mw4_year_cohort": (
        "Consider the choices that carry period style: the key and whether it "
        "is major or minor, the metre and its character, the tempo or movement "
        "type, and the formal type (such as da capo aria). For each, look at "
        "what the other arias of that year mostly do -- the prevailing "
        "convention -- and whether this aria follows it or departs from it."
    ),
}

TIER = {
    "sw1_modulation": "lookup", "sw2_cadences": "inference", "sw3_style": "inference",
    "sw4_strophic": "inference", "sw5_affect": "inference-light",
    "mw1_period_norm": "retrieval-inference", "mw2_bsection_opening": "retrieval-inference",
    "mw3_composer_textplan": "retrieval-comparison", "mw4_year_cohort": "retrieval-aggregation",
}

# which arias get which questions ----------------------------------------
CLUSTER_A = ["0001", "0012", "0041"]   # Son regina e sono amante
CLUSTER_B = ["0063", "0188", "0246"]   # Fra lo splendor del trono
CLUSTER_C = ["0092", "0101", "0112"]   # Se resto sul lido
MINORS = ["0100", "0217", "2354"]
ALL12 = CLUSTER_A + CLUSTER_B + CLUSTER_C + MINORS

ASSIGN = {
    "sw1_modulation": ALL12,
    "sw2_cadences": ALL12,
    "sw3_style": ["0001", "0012", "0246", "0100", "0217", "2354"],
    "sw4_strophic": ["0001", "0092", "0246", "2354"],
    "sw5_affect": ["0217", "0100", "0041", "0246"],
    "mw1_period_norm": CLUSTER_A + CLUSTER_B + CLUSTER_C,
    "mw2_bsection_opening": ["0041", "0188", "0112"],
    "mw3_composer_textplan": ["0001", "0012", "0041"],
    "mw4_year_cohort": ["0012", "0246", "0112"],
}

CORE = {"sw1_modulation", "sw2_cadences", "mw1_period_norm"}


def main() -> None:
    assert Q_BASE.keys() == Q_EXPERT.keys() == ASSIGN.keys() == TIER.keys(), (
        "Q_BASE / Q_EXPERT / ASSIGN / TIER must cover the same question templates"
    )

    meta = {json.loads(l)["record_id"]: json.loads(l)
            for l in SAMPLE.read_text().splitlines() if l.strip()}
    rows = []
    for qid, recs in ASSIGN.items():
        for rid in recs:
            m = meta[rid]
            ref = (f"record_id {rid} ({m['composer']}'s {m['year']} setting of "
                   f"'{m['aria_name']}')")
            rows.append({
                "task_id": f"{qid}__{rid}",
                "query": Q_BASE[qid].format(ref=ref),
                "category": qid,
                "tags": [TIER[qid]] + (["core"] if qid in CORE else [])
                        + (["multi_work"] if qid.startswith("mw") else ["single_work"]),
                "metadata": {
                    "record_id": rid, "question_template": qid,
                    "aria_name": m["aria_name"], "composer": m["composer"],
                    "year": m["year"], "decade": m["decade"],
                    "initial_key": m["initial_key"], "initial_meter": m["initial_meter"],
                },
            })
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {len(rows)} tasks to {OUT.relative_to(REPO)}")
    by = {}
    for r in rows:
        by[r["category"]] = by.get(r["category"], 0) + 1
    for k, v in by.items():
        print(f"  {k:24} {v}")

    EXPERT_OUT.write_text(json.dumps(dict(Q_EXPERT), ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(Q_EXPERT)} expert blocks to {EXPERT_OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
