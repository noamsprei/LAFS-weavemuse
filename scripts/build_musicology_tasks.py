"""Generate the musicology study's dataset + expert-prompt file from the
frozen 12-aria sample.

Writes two artifacts, deterministically (edit the tables and rerun):

  data/eval/tasks_musicology.jsonl        -- one EvalTask per line, `query`
                                             built from Q_BASE (the neutral
                                             question every condition gets)
  data/eval/expert_prompts_musicology.json -- {question_template -> decomposition
                                             block} from Q_EXPERT, plus a
                                             "_shared" evidence-first preamble

The study's two conditions:
  default -> the task's base query, verbatim.
  expert  -> base query + "_shared" preamble + the Q_EXPERT block for that
             question template (appended by weavemuse/eval/runner.py at run
             time, keyed on the task's `category`).

Split rule (see weavemuse/eval/STUDY_LOG.md):
  - Q_BASE carries the bare question, the deliverables, and -- for the four
    templates with a computed reference answer (sw1, sw2, sw4, mw1) -- the
    definition of the target concept, so the reference and the answer are
    judged on the same criterion regardless of condition.
  - Q_EXPERT adds ONLY the analytical decomposition (how a musicologist breaks
    the question down and what counts as evidence) and, for the open style/
    affect questions, the relevant feature bundles. It names no tool and no
    sub-agent -- mapping a concept onto a capability is the agent's job and
    part of what is scored.

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
# Every study condition gets exactly this text; the expert condition gets the
# matching Q_EXPERT block appended.
Q_BASE = {
    "sw1_modulation": (
        "For the Didone-corpus aria {ref}: give its tonal scheme -- the ordered "
        "sequence of key areas -- and then count the CONFIRMED modulations. Count "
        "a key area as a confirmed modulation only if the music establishes it "
        "with its own cadence (a phrase closing with a perfect or half cadence in "
        "that key); a region of only a few bars that is passed through without a "
        "cadential close is a tonicisation and does NOT count. The da capo return "
        "to the opening key is a return, not a new modulation. Report: the ordered "
        "key areas, which of them are confirmed modulations (with the cadential "
        "evidence), and the total confirmed count."
    ),
    "sw2_cadences": (
        "For the Didone-corpus aria {ref}: how many cadences does it contain, and "
        "are most of them of the same type? A cadence is the harmonic close that "
        "ends a phrase; classify each by the progression into it -- perfect "
        "(dominant to tonic), imperfect (a phrase ending on the dominant), "
        "interrupted (dominant to submediant), or plagal (subdominant to tonic). "
        "Report the total count, the type of each, and whether one type "
        "predominates."
    ),
    "sw3_style": (
        "For the Didone-corpus aria {ref}: is it more galant or more Baroque in "
        "character? Decide which features are relevant, examine them, and weigh "
        "them."
    ),
    "sw4_strophic": (
        "For the Didone-corpus aria {ref}: infer its strophic structure -- the "
        "ordered sequence of A and B sections -- from the music alone, without "
        "access to its sung text. Take an A section to be a statement of the "
        "opening material in the home key, and a B section to be a contrasting "
        "passage that is framed away from the tonic. Report the ordered sequence "
        "and explain the reasoning."
    ),
    "sw5_affect": (
        "For the Didone-corpus aria {ref}: is it stormy (Sturm und Drang) or "
        "calm and pastoral in character? Decide which features bear on this, "
        "examine them, and weigh them."
    ),
    "mw1_period_norm": (
        "For the Didone-corpus aria {ref}: is its tonal scheme typical of arias "
        "written in the same period? A feature is typical only in relation to a "
        "body of contemporaneous works, so base the answer on the distribution "
        "across a representative sample of arias from the same time, not on this "
        "aria alone. Report the comparison and the judgement."
    ),
    "mw2_bsection_opening": (
        "For the Didone-corpus aria {ref}: is the chord progression that opens "
        "its B section common among other settings of the same aria text? "
        "Compare against the other settings before answering."
    ),
    "mw3_composer_textplan": (
        "For the Didone-corpus aria {ref}: does its strophic/formal plan "
        "resemble that of other arias by the same composer? Compare against a "
        "sample of that composer's other arias."
    ),
    "mw4_year_cohort": (
        "For the Didone-corpus aria {ref}: do most arias composed in the same "
        "year share similar musical characteristics (key, meter, tempo, form)? "
        "Compare against the other arias of that year."
    ),
}

# expert decomposition blocks ------------------------------------------------
# Appended (after EXPERT_SHARED) to the base query only in the `expert`
# condition. Method and evidence only -- no tool or sub-agent is named.
EXPERT_SHARED = (
    "Work from specific musical evidence, not general impression. Break the "
    "question into concrete sub-questions that each have an evidential answer, "
    "gather that evidence, then reason to a conclusion and state the evidence it "
    "rests on."
)

Q_EXPERT = {
    "sw1_modulation": (
        "To work this out: first establish the home key. Then go through the "
        "tonal design section by section; for each section, the harmonic function "
        "it closes on is the cadential evidence for whether a new key was "
        "confirmed there. Count only the confirmed key changes and give the "
        "ordered sequence of key areas. 'No modulation beyond the dominant' is a "
        "legitimate finding."
    ),
    "sw2_cadences": (
        "To find the cadences, locate the points of harmonic arrival or repose -- "
        "where a phrase comes to rest -- and read the chord progression leading "
        "into each; that progression is the cadence type. Then tally the types "
        "and see whether one predominates."
    ),
    "sw3_style": (
        "Galant and Baroque character are bundles of tendencies, not single "
        "traits. Baroque writing tends toward continuous spinning-out of the "
        "line, denser counterpoint, motoric and consistent rhythm, and few full "
        "stops. Galant writing tends toward periodic phrasing with frequent clear "
        "cadences, lighter melody-and-accompaniment texture, and more sectional "
        "articulation. First decide which of these features you can actually check "
        "from the evidence available, then weigh those to decide which pole the "
        "aria leans toward."
    ),
    "sw4_strophic": (
        "The stanzas of the text usually leave a musical trace. Locate the "
        "repeats by finding where the opening tonal motion is restated exactly; a "
        "contrasting passage that begins and ends away from the tonic, sitting "
        "between such restatements, is the middle strophe."
    ),
    "sw5_affect": (
        "Stormy (Sturm und Drang) writing tends toward fast tempo, minor mode, "
        "dense and agitated motion, and driving rhythm. Pastoral writing tends "
        "toward moderate or slow tempo, a lilting compound or triple metre, "
        "transparent texture, and gentle motion. Decide which of these features "
        "you can check from the evidence, then weigh them rather than deciding on "
        "any one alone."
    ),
    "mw1_period_norm": (
        "To judge typicality: characterise this aria's tonal scheme -- home key, "
        "the ordered key areas, whether it stays within closely related keys. "
        "Assemble the same characterisation for a sample of arias from the same "
        "period. Compare this aria against that distribution -- common, at the "
        "edge, or unusual. One aria cannot establish a norm."
    ),
    "mw2_bsection_opening": (
        "The B section is the contrasting middle of the aria, framed away from "
        "the home key. Locate where it begins, read the opening chord progression "
        "there, then obtain the corresponding opening progression for the other "
        "settings of the same text and compare -- is this opening gesture shared "
        "across settings or particular to this one?"
    ),
    "mw3_composer_textplan": (
        "A formal plan here is the sequence of sections -- A and B strophes and "
        "any da capo return -- together with their tonal framing. Describe this "
        "aria's plan, then the plans of a sample of the same composer's other "
        "arias, and compare: is this one characteristic of how the composer "
        "builds an aria, or an outlier?"
    ),
    "mw4_year_cohort": (
        "Collect the stated characteristics -- key, metre, tempo, form -- for "
        "this aria and for the other arias of the same year, then look at the "
        "spread of each: is there a dominant key, metre, tempo, or formal type "
        "that year, and does this aria fall inside it? A shared characteristic is "
        "a concentration in the distribution, not just two arias that happen to "
        "match."
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

    expert = {"_shared": EXPERT_SHARED, **Q_EXPERT}
    EXPERT_OUT.write_text(json.dumps(expert, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(Q_EXPERT)} expert blocks (+ _shared) to "
          f"{EXPERT_OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
