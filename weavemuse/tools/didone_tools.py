"""Didone-corpus query tools for the musicology-analysis agent.

These tools expose the pre-computed structured analyses of the Didone dataset
(18th-century Italian opera arias -- multiple composers setting the same
libretto texts across ~1720-1800) as smolagents tools. They are read-only
CSV lookups: no model weights, no GPU, no MIDI parsing at call time.

Every tool returns a JSON string. Read the parsed structure directly -- do not
write string-splitting code to pull fields out of it.

Data location is resolved from ``$DIDONE_DATA_DIR`` (default: the repo root),
which must contain::

    metadata.csv
    harmony_analysis/tonal_plan_overview.csv
    harmony_analysis/tonal_plan_segments.csv
    harmony_analysis/parsed_harmony_events.csv
    text_tonal_alignment/section_tonal_plan.csv
    textual_plan/textual_plan_overview.csv
    textual_plan/textual_plan_sections.csv

Design note for the prompt-intervention study: these tools deliberately return
*evidence*, not *conclusions*. There is no ``get_cadences`` or ``get_style``
tool -- the agent is expected to call ``get_harmony`` over a measure range and
infer cadences / harmonic rhythm / style itself. That inference is the thing
the study measures.
"""

from __future__ import annotations

import json
import math
import os
import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from smolagents import Tool

# --- data access -----------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]

_FILES = {
    "metadata": "metadata.csv",
    "tonal_plan_overview": "harmony_analysis/tonal_plan_overview.csv",
    "tonal_plan_segments": "harmony_analysis/tonal_plan_segments.csv",
    "harmony_events": "harmony_analysis/parsed_harmony_events.csv",
    "section_tonal_plan": "text_tonal_alignment/section_tonal_plan.csv",
    "textual_plan_overview": "textual_plan/textual_plan_overview.csv",
    "textual_plan_sections": "textual_plan/textual_plan_sections.csv",
}

# One row per record_id expected; a few duplicate ids exist in the source
# (near-identical file variants) and are collapsed to the first on load.
_SINGLE_ROW = {"metadata", "tonal_plan_overview", "textual_plan_overview"}

# harmony_events is ~190MB; only these columns are needed by get_harmony.
_HARMONY_COLS = [
    "record_id", "measure_number", "beat", "normalized_label", "local_function",
    "tonal_region", "absolute_key", "is_region_marker", "is_secondary_dominant",
    "applied_to", "inversion", "quality_or_chord_type", "phrase_start", "phrase_end",
]

_MAX_HARMONY_SPAN = 40  # measures per get_harmony call


def _data_dir() -> Path:
    return Path(os.getenv("DIDONE_DATA_DIR", str(_REPO_ROOT)))


@lru_cache(maxsize=None)
def _load(name: str) -> pd.DataFrame:
    path = _data_dir() / _FILES[name]
    if not path.exists():
        raise FileNotFoundError(
            f"Didone data file not found: {path}. Set $DIDONE_DATA_DIR to the "
            f"directory containing metadata.csv and the harmony_analysis/, "
            f"text_tonal_alignment/, textual_plan/ subfolders."
        )
    kwargs: dict = {"dtype": {"record_id": str}}
    if name == "harmony_events":
        kwargs["usecols"] = [c for c in _HARMONY_COLS]
    df = pd.read_csv(path, **kwargs)
    df["record_id"] = df["record_id"].astype(str).str.strip()
    if name in _SINGLE_ROW:
        df = df.drop_duplicates(subset="record_id", keep="first").reset_index(drop=True)
    return df


def _row(name: str, record_id: str) -> pd.Series | None:
    df = _load(name)
    hit = df[df["record_id"] == str(record_id).strip()]
    return None if hit.empty else hit.iloc[0]


def _clean(v):
    """pandas/NaN -> JSON-safe scalar."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, str) and v.strip() == "":
        return None
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
    except Exception:
        pass
    return v


def _records(df: pd.DataFrame, cols: list[str], rename: dict | None = None) -> list[dict]:
    rename = rename or {}
    out = []
    for _, r in df.iterrows():
        out.append({rename.get(c, c): _clean(r[c]) for c in cols})
    return out


def _dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, default=str)


_BARE_KEY_RE = re.compile(r"^[A-Ga-g][b#]?$")


def _parse_home(initial_key: str | None):
    """'A major' / 'Bb minor' -> ('A', 'major'). None if unparseable."""
    if not initial_key:
        return None
    m = re.match(r"\s*([A-Ga-g])([b#]?)\s*(major|minor|maj|min)?", str(initial_key))
    if not m:
        return None
    tonic = m.group(1).upper() + m.group(2)
    mode = (m.group(3) or "major").lower()
    mode = "minor" if mode.startswith("min") else "major"
    return tonic, mode


@lru_cache(maxsize=4096)
def _abs_key(home_initial_key: str | None, region: str | None) -> str | None:
    """Absolute key of a tonal region given the aria's home key and the region's
    Roman-numeral label (e.g. home 'A major' + region 'V' -> 'E major';
    'vi' -> 'F# minor'). Returns None if it can't be resolved.
    """
    if not region:
        return None
    region = str(region).strip()
    home = _parse_home(home_initial_key)
    if _BARE_KEY_RE.match(region):
        # segment labelled with a bare key letter (usually segment 1 == home);
        # letter case carries the mode (lower = minor)
        letter = region[0].upper() + region[1:]
        mode = "minor" if region[0].islower() else "major"
        if home and letter.rstrip("b#") == home[0].rstrip("b#"):
            return f"{home[0]} {home[1]}"
        return f"{letter} {mode}"
    if not home:
        return None
    try:
        from music21 import key as m21key
        from music21 import roman
        k = m21key.Key(home[0], home[1])
        rn = roman.RomanNumeral(region, k)
        root = rn.root().name.replace("-", "b")
        mode = "minor" if region[:1].islower() else "major"
        return f"{root} {mode}"
    except Exception:
        return None


# --- tools ---------------------------------------------------------------

class DidoneCorpusSearchTool(Tool):
    name = "search_didone_corpus"
    description = (
        "Search the Didone aria corpus (18th-c. Italian opera arias) by any "
        "combination of filters and get back a list of matching arias with their "
        "record_id and basic metadata. Use this to build a comparison sample -- "
        "e.g. every aria from the 1740s, or every setting by a given composer, or "
        "every setting of the same aria text. Returns a JSON object "
        "{n_matches, arias:[{record_id, aria_name, composer, year, decade, "
        "initial_key, initial_meter, initial_tempo}]}. Read the JSON directly."
    )
    inputs = {
        "aria_name": {"type": "string", "description": "Exact aria text/title, e.g. 'Se resto sul lido'. Case-insensitive.", "nullable": True},
        "composer": {"type": "string", "description": "Composer name substring, e.g. 'Hasse'. Case-insensitive.", "nullable": True},
        "decade": {"type": "integer", "description": "Decade as a 4-digit year, e.g. 1740.", "nullable": True},
        "year_from": {"type": "integer", "description": "Earliest year of composition (inclusive).", "nullable": True},
        "year_to": {"type": "integer", "description": "Latest year of composition (inclusive).", "nullable": True},
        "initial_key": {"type": "string", "description": "Opening key, e.g. 'G minor' or just 'minor'. Case-insensitive substring.", "nullable": True},
        "limit": {"type": "integer", "description": "Max rows to return (default 60).", "nullable": True},
    }
    output_type = "string"

    def forward(self, aria_name=None, composer=None, decade=None, year_from=None,
                year_to=None, initial_key=None, limit=None) -> str:
        df = _load("metadata").copy()
        df["_year"] = pd.to_numeric(df["year_of_composition"], errors="coerce")
        if aria_name:
            df = df[df["aria_name"].str.strip().str.lower() == aria_name.strip().lower()]
        if composer:
            df = df[df["composer"].str.contains(composer, case=False, na=False)]
        if decade is not None:
            df = df[pd.to_numeric(df["decade"], errors="coerce") == int(decade)]
        if year_from is not None:
            df = df[df["_year"] >= float(year_from)]
        if year_to is not None:
            df = df[df["_year"] <= float(year_to)]
        if initial_key:
            df = df[df["initial_key"].str.contains(initial_key, case=False, na=False)]
        df = df.sort_values(["_year", "composer"])
        total = len(df)
        df = df.head(int(limit) if limit else 60)
        cols = ["record_id", "aria_name", "composer", "year_of_composition",
                "decade", "initial_key", "initial_meter", "initial_tempo_text"]
        rename = {"year_of_composition": "year", "initial_tempo_text": "initial_tempo"}
        return _dump({
            "n_matches": total,
            "returned": len(df),
            "arias": _records(df, cols, rename),
        })


class DidoneMetadataTool(Tool):
    name = "get_aria_metadata"
    description = (
        "Get score-level metadata for one aria by record_id: composer, year, "
        "opera, character, opening key and key-change count, meter (and whether "
        "it changes), tempo marking and tempo family, da capo / repeat scheme, "
        "measure count, note density, lyric syllable density, and instrumentation. "
        "Returns a JSON object. Does NOT contain the harmonic analysis -- use "
        "get_tonal_plan / get_harmony for that."
    )
    inputs = {"record_id": {"type": "string", "description": "Aria record_id, e.g. '0012'."}}
    output_type = "string"

    def forward(self, record_id: str) -> str:
        r = _row("metadata", record_id)
        if r is None:
            return _dump({"error": f"no aria with record_id {record_id!r}"})
        cols = [
            "record_id", "aria_name", "composer", "year_of_composition", "decade",
            "opera_name", "character", "initial_key", "n_key_signature_changes",
            "initial_meter", "meter_family", "n_meter_changes", "unique_meters",
            "initial_tempo_text", "tempo_family", "n_tempo_changes",
            "repeat_type", "has_segno", "has_coda", "has_fine",
            "n_measures_written", "vocal_measures_count", "first_vocal_measure",
            "last_vocal_measure", "note_density_per_measure", "n_notes", "n_rests",
            "n_tuplets", "lyric_density_per_measure", "n_lyric_syllables",
            "harmony_annotation_density", "n_harmony_annotations",
            "instrumentation_signature",
        ]
        return _dump({c: _clean(r[c]) for c in cols})


class DidoneTonalPlanTool(Tool):
    name = "get_tonal_plan"
    description = (
        "Get the tonal plan of one aria. Returns a JSON object with: tonal_plan "
        "(the ordered key-area string), home_key, and segments -- a list where "
        "each entry gives the measure span, the region as a Roman numeral "
        "relative to the home key, the absolute key of that region, the harmonic "
        "function the segment opens and closes on (closes_on is your cadence "
        "evidence -- whether the segment ends with a cadential close IN that "
        "region), an annotation boundary_type, and a confidence score. Read the "
        "JSON directly; do not string-split it. Whether a short region counts as "
        "a real modulation or just a passing tonicisation is for you to judge "
        "from its length and closes_on -- the tool does not label that."
    )
    inputs = {"record_id": {"type": "string", "description": "Aria record_id, e.g. '0012'."}}
    output_type = "string"

    def forward(self, record_id: str) -> str:
        ov = _row("tonal_plan_overview", record_id)
        md = _row("metadata", record_id)
        seg = _load("tonal_plan_segments")
        seg = seg[seg["record_id"] == str(record_id).strip()].sort_values("segment_index")
        if ov is None and seg.empty:
            return _dump({"error": f"no tonal plan for record_id {record_id!r}"})

        home_initial = _clean(md["initial_key"]) if md is not None else None
        if not home_initial and not seg.empty:
            home_initial = _clean(seg.iloc[0]["absolute_key"])

        segments = []
        for _, r in seg.iterrows():
            region = _clean(r["region_label"])
            key = _abs_key(home_initial, region) or _clean(r["absolute_key"])
            try:
                span = int(_clean(r["end_measure"])) - int(_clean(r["start_measure"]))
            except (TypeError, ValueError):
                span = None
            segments.append({
                "segment": _clean(r["segment_index"]),
                "measures": f"{_clean(r['start_measure'])}-{_clean(r['end_measure'])}",
                "length_measures": span,
                "region": region,
                "key": key,
                "opens_on": _clean(r["opening_function"]),
                "closes_on": _clean(r["closing_function"]),
                "boundary_type": _clean(r["modulation_type"]),
                "confidence": _clean(r["confidence"]),
            })

        return _dump({
            "record_id": str(record_id).strip(),
            "tonal_plan": _clean(ov["tonal_plan"]) if ov is not None else None,
            "home_key": home_initial or (segments[0]["key"] if segments else None),
            "n_segments": len(segments),
            "segments": segments,
        })


class DidoneHarmonyTool(Tool):
    name = "get_harmony"
    description = (
        "Get the chord-by-chord Roman-numeral analysis of one aria over a "
        "measure range. Returns a JSON object {record_id, from_measure, "
        "to_measure, n_events, chords:[{measure, beat, label, function, "
        "tonal_region, secondary_dominant_of, inversion, quality, phrase_start, "
        "phrase_end}]}. There is NO cadence label -- infer cadences yourself from "
        "the progression into each phrase_end. Read the JSON directly. Max "
        f"{_MAX_HARMONY_SPAN} measures per call; page a long aria with several calls."
    )
    inputs = {
        "record_id": {"type": "string", "description": "Aria record_id, e.g. '0012'."},
        "from_measure": {"type": "integer", "description": "First measure of the range (inclusive)."},
        "to_measure": {"type": "integer", "description": f"Last measure (inclusive). Must be within {_MAX_HARMONY_SPAN} of from_measure."},
    }
    output_type = "string"

    def forward(self, record_id: str, from_measure: int, to_measure: int) -> str:
        a, b = int(from_measure), int(to_measure)
        if b < a:
            a, b = b, a
        if b - a > _MAX_HARMONY_SPAN:
            return _dump({"error": f"range too wide ({b - a} measures); ask for at most "
                                   f"{_MAX_HARMONY_SPAN} measures per call"})
        df = _load("harmony_events")
        mnum = pd.to_numeric(df["measure_number"], errors="coerce")
        df = df[(df["record_id"] == str(record_id).strip()) & (mnum >= a) & (mnum <= b)]
        if df.empty:
            return _dump({"error": f"no harmony events for record_id {record_id!r} in mm. {a}-{b}"})
        chords = []
        for _, r in df.iterrows():
            chords.append({
                "measure": _clean(r["measure_number"]),
                "beat": _clean(r["beat"]),
                "label": _clean(r["normalized_label"]),
                "function": _clean(r["local_function"]),
                "tonal_region": _clean(r["tonal_region"]),
                "secondary_dominant_of": _clean(r["applied_to"]) if _clean(r["is_secondary_dominant"]) else None,
                "inversion": _clean(r["inversion"]),
                "quality": _clean(r["quality_or_chord_type"]),
                "phrase_start": bool(_clean(r["phrase_start"])),
                "phrase_end": bool(_clean(r["phrase_end"])),
            })
        return _dump({
            "record_id": str(record_id).strip(),
            "from_measure": a, "to_measure": b, "n_events": len(chords),
            "chords": chords,
        })


class DidoneSectionTonalPlanTool(Tool):
    name = "get_section_tonal_plan"
    description = (
        "Get the formal sections of one aria aligned with their local tonal "
        "plans. Returns a JSON object {record_id, written_plan, performed_plan, "
        "sections:[{index, symbol, type, measures, section_tonal_plan, "
        "dominant_region}]}. Read the JSON directly. Use this for questions about "
        "strophic/formal structure and where the tonal plan repeats (da capo)."
    )
    inputs = {"record_id": {"type": "string", "description": "Aria record_id, e.g. '0012'."}}
    output_type = "string"

    def forward(self, record_id: str) -> str:
        df = _load("section_tonal_plan")
        df = df[df["record_id"] == str(record_id).strip()].sort_values("section_index")
        if df.empty:
            return _dump({"error": f"no section tonal plan for record_id {record_id!r}"})
        r0 = df.iloc[0]
        sections = []
        for _, r in df.iterrows():
            sections.append({
                "index": _clean(r["section_index"]),
                "symbol": _clean(r["section_symbol"]),
                "type": _clean(r["section_type"]),
                "measures": f"{_clean(r['section_start_measure'])}-{_clean(r['section_end_measure'])}",
                "section_tonal_plan": _clean(r["section_tonal_plan"]),
                "dominant_region": _clean(r["dominant_tonal_region"]),
            })
        return _dump({
            "record_id": str(record_id).strip(),
            "written_plan": _clean(r0["written_plan"]),
            "performed_plan": _clean(r0["performed_plan"]),
            "sections": sections,
        })


class DidoneTextStructureTool(Tool):
    name = "get_text_structure"
    description = (
        "Get the textual/strophic structure of one aria. Returns a JSON object "
        "{record_id, written_plan, repeat_scheme, performed_plan, "
        "text_confidence, sections:[{index, symbol, type, measures, "
        "text_preview}]}. Read the JSON directly. Use this for questions about "
        "strophic form and text setting."
    )
    inputs = {"record_id": {"type": "string", "description": "Aria record_id, e.g. '0012'."}}
    output_type = "string"

    def forward(self, record_id: str) -> str:
        ov = _row("textual_plan_overview", record_id)
        sec = _load("textual_plan_sections")
        sec = sec[sec["record_id"] == str(record_id).strip()].sort_values("section_index")
        if ov is None and sec.empty:
            return _dump({"error": f"no textual structure for record_id {record_id!r}"})
        sections = []
        for _, r in sec.iterrows():
            sections.append({
                "index": _clean(r["section_index"]),
                "symbol": _clean(r["symbol"]),
                "type": _clean(r["section_type"]),
                "measures": f"{_clean(r['start_measure'])}-{_clean(r['end_measure'])}",
                "text_preview": _clean(r["text_preview"]),
            })
        return _dump({
            "record_id": str(record_id).strip(),
            "written_plan": _clean(ov["written_plan"]) if ov is not None else None,
            "repeat_scheme": _clean(ov["repeat_scheme"]) if ov is not None else None,
            "performed_plan": _clean(ov["performed_plan"]) if ov is not None else None,
            "text_confidence": _clean(ov["text_confidence_status"]) if ov is not None else None,
            "sections": sections,
        })


_MAX_BATCH = 80


def _parse_ids(record_ids: str) -> list[str]:
    return [x.strip() for x in str(record_ids).replace(",", " ").split() if x.strip()]


class DidoneTonalPlansBatchTool(Tool):
    name = "get_tonal_plans"
    description = (
        "Get the tonal-plan string and opening key for MANY arias at once, given "
        "a list of record_ids. Use this after search_didone_corpus to compare an "
        "aria against a sample of its peers. Returns a JSON object "
        "{n, arias:[{record_id, composer, year, initial_key, tonal_plan}], "
        "missing:[...]}. Read the JSON directly. Max 80 ids."
    )
    inputs = {"record_ids": {"type": "string", "description": "record_ids separated by commas or spaces, e.g. '0001, 0012, 0041'."}}
    output_type = "string"

    def forward(self, record_ids: str) -> str:
        ids = _parse_ids(record_ids)[:_MAX_BATCH]
        if not ids:
            return _dump({"error": "no record_ids given"})
        ov = _load("tonal_plan_overview")
        md = _load("metadata").set_index("record_id")
        sub = ov[ov["record_id"].isin(ids)].copy()
        if sub.empty:
            return _dump({"n": 0, "arias": [], "missing": ids})
        sub["initial_key"] = sub["record_id"].map(md["initial_key"])
        sub["year"] = sub["record_id"].map(md["year_of_composition"])
        cols = ["record_id", "composer_name", "year", "initial_key", "tonal_plan"]
        rename = {"composer_name": "composer"}
        found = set(sub["record_id"])
        return _dump({
            "n": len(sub),
            "arias": _records(sub, cols, rename),
            "missing": [i for i in ids if i not in found],
        })


class DidoneSectionTonalPlansBatchTool(Tool):
    name = "get_section_tonal_plans"
    description = (
        "Get the per-section tonal plans (R1/A1/B1... with each section's local "
        "key sequence and measure span) for MANY arias at once. Use this to "
        "compare formal/tonal design -- e.g. how the B section opens -- across a "
        "sample of arias. Returns a JSON object {n_arias, by_record:{record_id: "
        "[{symbol, type, measures, section_tonal_plan}]}}. Read the JSON "
        "directly. Max 80 ids."
    )
    inputs = {"record_ids": {"type": "string", "description": "record_ids separated by commas or spaces."}}
    output_type = "string"

    def forward(self, record_ids: str) -> str:
        ids = _parse_ids(record_ids)[:_MAX_BATCH]
        if not ids:
            return _dump({"error": "no record_ids given"})
        df = _load("section_tonal_plan")
        sub = df[df["record_id"].isin(ids)].sort_values(["record_id", "section_index"])
        if sub.empty:
            return _dump({"n_arias": 0, "by_record": {}})
        by_record: dict[str, list] = {}
        for _, r in sub.iterrows():
            by_record.setdefault(_clean(r["record_id"]), []).append({
                "symbol": _clean(r["section_symbol"]),
                "type": _clean(r["section_type"]),
                "measures": f"{_clean(r['section_start_measure'])}-{_clean(r['section_end_measure'])}",
                "section_tonal_plan": _clean(r["section_tonal_plan"]),
            })
        return _dump({"n_arias": len(by_record), "by_record": by_record})


class DidoneTextStructuresBatchTool(Tool):
    name = "get_text_structures"
    description = (
        "Get the strophic/formal structure (written plan, repeat scheme, "
        "performed plan) for MANY arias at once. Use this to compare an aria's "
        "text setting against other arias by the same composer. Returns a JSON "
        "object {n, arias:[{record_id, aria_name, written_plan, repeat_scheme, "
        "performed_plan}]}. Read the JSON directly. Max 80 ids."
    )
    inputs = {"record_ids": {"type": "string", "description": "record_ids separated by commas or spaces."}}
    output_type = "string"

    def forward(self, record_ids: str) -> str:
        ids = _parse_ids(record_ids)[:_MAX_BATCH]
        if not ids:
            return _dump({"error": "no record_ids given"})
        df = _load("textual_plan_overview")
        sub = df[df["record_id"].isin(ids)]
        if sub.empty:
            return _dump({"n": 0, "arias": []})
        cols = ["record_id", "aria_name", "written_plan", "repeat_scheme", "performed_plan"]
        return _dump({"n": len(sub), "arias": _records(sub, cols)})


def didone_tools() -> list[Tool]:
    """All Didone query tools, freshly instantiated."""
    return [
        DidoneCorpusSearchTool(),
        DidoneMetadataTool(),
        DidoneTonalPlanTool(),
        DidoneHarmonyTool(),
        DidoneSectionTonalPlanTool(),
        DidoneTextStructureTool(),
        DidoneTonalPlansBatchTool(),
        DidoneSectionTonalPlansBatchTool(),
        DidoneTextStructuresBatchTool(),
    ]
