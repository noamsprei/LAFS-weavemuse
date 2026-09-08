"""Compute domain-expert reference answers for the scriptable question types
and write an augmented dataset the judge can score against.

Reads  data/eval/tasks_musicology.jsonl
Writes data/eval/tasks_musicology.local.jsonl   (gitignored -- derived from the
       confidential corpus) with metadata.reference filled in for
       sw1_modulation, sw2_cadences, sw4_strophic, mw1_period_norm.

The subjective question types (sw3_style, sw5_affect, mw2/mw3) get no reference
here -- those stay for human calibration grading.

Run where the Didone CSVs are reachable ($DIDONE_DATA_DIR). Uses the same
DidoneTool query logic as the agent, so a reference reflects exactly what the
agent could have retrieved.
"""

from __future__ import annotations

import json
import os
import sys
import types
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC = REPO / "data" / "eval" / "tasks_musicology.jsonl"
OUT = REPO / "data" / "eval" / "tasks_musicology.local.jsonl"

# import didone_tools without pulling smolagents (only the Tool base class needs it)
if "smolagents" not in sys.modules:
    _stub = types.ModuleType("smolagents")
    _stub.Tool = object
    sys.modules["smolagents"] = _stub

sys.path.insert(0, str(REPO))
from weavemuse.tools import didone_tools as dt  # noqa: E402

_TONAL = dt.DidoneTonalPlanTool()
_HARM = dt.DidoneHarmonyTool()
_SECT = dt.DidoneSectionTonalPlanTool()

_TONIC_CLOSE = {"I", "i", "I(64)", "i(64)"}
_HALF_CLOSE = {"V", "V7", "V(64)", "V7(64)", "V65", "V6"}
_MIN_CONFIRM_BARS = 8


def ref_sw1(rid: str) -> str:
    tp = json.loads(_TONAL.forward(rid))
    if tp.get("error"):
        return f"unavailable: {tp['error']}"
    segs = tp["segments"]
    home = tp.get("home_key")
    home_letter = (home or "").split()[0] if home else None
    lines = [f"Home key: {home}.", f"Tonal-plan string: {tp.get('tonal_plan')}.",
             "Segments (region / key / bars / closes_on):"]
    if segs and home_letter and str(segs[0]["key"]).split()[0] not in (home_letter, "None"):
        lines.append(f"NOTE: metadata home key ({home}) and the tonal-plan opening "
                     f"region ({segs[0]['key']}) disagree -- the plan may be notated "
                     f"relative to a different tonic; weigh the region sequence, not "
                     f"the metadata key, if they conflict.")
    confirmed = []
    for s in segs:
        region, key, ln, closes = s["region"], s["key"], s["length_measures"], s["closes_on"]
        lines.append(f"  {region} / {key} / {ln} bars / closes_on {closes}")
        is_home = region in ("I", "i") or (home_letter and str(key).split()[0] == home_letter)
        if is_home:
            continue
        long_enough = (ln or 0) >= _MIN_CONFIRM_BARS
        cadential = closes in _TONIC_CLOSE or closes in _HALF_CLOSE
        if long_enough and cadential:
            confirmed.append(f"{key} ({region}, {ln} bars, closes on {closes})")
    lines.append(
        f"Confirmed modulations (>= {_MIN_CONFIRM_BARS} bars AND a cadential close "
        f"in that key): {len(confirmed)}"
        + (": " + "; ".join(confirmed) if confirmed else "")
    )
    lines.append("Short regions with no cadential close are tonicisations, not "
                 "modulations. The da capo return to the home key is a return.")
    return "\n".join(lines)


def _classify_cadence(prev_fn: str | None, fn: str | None) -> str | None:
    if not fn:
        return None
    f = fn.upper()
    p = (prev_fn or "").upper()
    if f.startswith("I") and (p.startswith("V")):
        return "perfect"
    if f.startswith("VI") and p.startswith("V"):
        return "interrupted"
    if f.startswith("I") and p.startswith("IV"):
        return "plagal"
    if f.startswith("V"):
        return "half"
    return "other"


def ref_sw2(rid: str) -> str:
    tp = json.loads(_TONAL.forward(rid))
    if tp.get("error"):
        return f"unavailable: {tp['error']}"
    # page get_harmony over the whole aria
    last = tp.get("total_measures")
    if not last:
        try:
            last = max(int(s["end_measure"]) for s in tp["segments"]
                       if s.get("end_measure") is not None)
        except ValueError:
            last = 200
    chords: list[dict] = []
    m = 1
    while m <= last:
        hi = min(m + dt._MAX_HARMONY_SPAN - 1, last)
        h = json.loads(_HARM.forward(rid, m, hi))
        if not h.get("error"):
            chords.extend(h["chords"])
        m = hi + 1
    if not chords:
        return "unavailable: no harmony events"
    cadences = []
    for i, c in enumerate(chords):
        if c.get("phrase_end"):
            prev_fn = chords[i - 1]["function"] if i else None
            cadences.append((c["measure"], _classify_cadence(prev_fn, c["function"])))
    tally = Counter(t for _, t in cadences)
    listing = "; ".join(f"m{mm}:{t}" for mm, t in cadences) or "none detected"
    top = tally.most_common(1)[0][0] if tally else "n/a"
    return (f"{len(cadences)} phrase-end cadences detected. By type: {dict(tally)}. "
            f"Most common: {top}. Locations: {listing}. "
            f"(Heuristic from the chord into each phrase_end; PAC vs IAC not "
            f"distinguishable from the data.)")


def ref_sw4(rid: str) -> str:
    s = json.loads(_SECT.forward(rid))
    if s.get("error"):
        return f"unavailable: {s['error']}"
    syms = [sec["symbol"] for sec in s["sections"]]
    a_like = [x for x in syms if x and x[0] in ("A",)]
    b_like = [x for x in syms if x and x[0] in ("B",)]
    return (f"Written plan: {s.get('written_plan')}. Performed plan: {s.get('performed_plan')}. "
            f"Text/aria sections: {syms}. A-strophe statements: {a_like or 'none'}; "
            f"B-strophe: {b_like or 'none'}. Expected reading: the A material recurs in "
            f"the home key (each recurrence = a fresh A stanza); a contrasting section "
            f"beginning and ending away from the tonic = the B stanza.")


def ref_mw1(rid: str, tasks_by_id: dict) -> str:
    meta = tasks_by_id[f"mw1_period_norm__{rid}"]["metadata"]
    decade = int(float(meta["decade"]))
    md = dt._load("metadata")
    ov = dt._load("tonal_plan_overview")
    ids = md[dt.pd.to_numeric(md["decade"], errors="coerce") == decade]["record_id"]
    plans = ov[ov["record_id"].isin(set(ids))]["tonal_plan"].dropna().tolist()
    this = json.loads(_TONAL.forward(rid)).get("tonal_plan")
    shapes = Counter(p.split(" -> ")[0:4].__str__() for p in plans)
    common = shapes.most_common(3)
    return (f"Decade {decade}s sample: n={len(plans)} arias with a tonal plan. "
            f"Most common opening shapes (first 4 areas): {common}. "
            f"This aria's plan: {this}. Judge whether its opening and overall shape "
            f"sit with the bulk of the decade sample or stand out.")


BUILDERS = {
    "sw1_modulation": lambda rid, tbi: ref_sw1(rid),
    "sw2_cadences": lambda rid, tbi: ref_sw2(rid),
    "sw4_strophic": lambda rid, tbi: ref_sw4(rid),
    "mw1_period_norm": lambda rid, tbi: ref_mw1(rid, tbi),
}


def main() -> None:
    rows = [json.loads(l) for l in SRC.read_text().splitlines() if l.strip()]
    tasks_by_id = {r["task_id"]: r for r in rows}
    n = 0
    for r in rows:
        b = BUILDERS.get(r["category"])
        if not b:
            continue
        rid = r["metadata"]["record_id"]
        try:
            r["metadata"]["reference"] = b(rid, tasks_by_id)
            n += 1
        except Exception as e:  # noqa: BLE001
            r["metadata"]["reference"] = f"reference computation failed: {type(e).__name__}: {e}"
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"wrote {OUT.relative_to(REPO)}  ({n}/{len(rows)} tasks got a reference)")
    for r in rows[:3]:
        if r["metadata"].get("reference"):
            print("\n---", r["task_id"], "---\n" + r["metadata"]["reference"][:600])


if __name__ == "__main__":
    main()
