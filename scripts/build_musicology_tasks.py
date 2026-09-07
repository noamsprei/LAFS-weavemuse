"""Generate data/eval/tasks_musicology.jsonl from the frozen 12-aria sample.

Deterministic: edit the ASSIGN table and rerun. Each row is an EvalTask
(weavemuse/eval/dataset.py schema): task_id, query, category, tags, metadata.
No gold answers are embedded -- correctness is judged post hoc (LLM judge +
a ~5-item human calibration sample).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SAMPLE = REPO / "data" / "eval" / "sample_12.jsonl"
OUT = REPO / "data" / "eval" / "tasks_musicology.jsonl"

# question templates -------------------------------------------------------
# {ref} is filled with "record_id N (Composer's YEAR setting of 'ARIA')"
Q = {
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
        "For the Didone-corpus aria {ref}: how many cadences does it contain, "
        "and are most of them of the same type? Classify each cadence you find "
        "(perfect, imperfect, interrupted, or plagal)."
    ),
    "sw3_style": (
        "For the Didone-corpus aria {ref}: is it more galant or more Baroque in "
        "character? Decide which features are relevant, examine them, and weigh "
        "them."
    ),
    "sw4_strophic": (
        "For the Didone-corpus aria {ref}: can you infer its strophic structure "
        "(the sequence of A and B stanzas) even without access to its sung text? "
        "Explain the reasoning."
    ),
    "sw5_affect": (
        "For the Didone-corpus aria {ref}: is it stormy (Sturm und Drang) or "
        "calm and pastoral in character? Decide which features bear on this, "
        "examine them, and weigh them."
    ),
    "mw1_period_norm": (
        "For the Didone-corpus aria {ref}: is its tonal scheme typical of arias "
        "written in the same period? Compare it against a representative sample "
        "of contemporaneous arias before answering."
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
                "query": Q[qid].format(ref=ref),
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


if __name__ == "__main__":
    main()
