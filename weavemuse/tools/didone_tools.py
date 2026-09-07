"""Didone-corpus query tools for the musicology-analysis agent.

These tools expose the pre-computed structured analyses of the Didone dataset
(18th-century Italian opera arias -- multiple composers setting the same
libretto texts across ~1720-1800) as smolagents tools. They are read-only
CSV lookups: no model weights, no GPU, no MIDI parsing at call time.

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

import os
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
    return df


def _row(name: str, record_id: str) -> pd.Series | None:
    df = _load(name)
    hit = df[df["record_id"] == str(record_id).strip()]
    return None if hit.empty else hit.iloc[0]


def _fmt_table(df: pd.DataFrame, cols: list[str], limit: int | None = None) -> str:
    view = df[cols].head(limit) if limit else df[cols]
    if view.empty:
        return "(no rows)"
    return view.to_csv(index=False).strip()


# --- tools ---------------------------------------------------------------

class DidoneCorpusSearchTool(Tool):
    name = "search_didone_corpus"
    description = (
        "Search the Didone aria corpus (18th-c. Italian opera arias) by any "
        "combination of filters and get back a list of matching arias with their "
        "record_id and basic metadata. Use this to build a comparison sample -- "
        "e.g. every aria from the 1740s, or every setting by a given composer, or "
        "every setting of the same aria text. Returns CSV text: "
        "record_id, aria_name, composer, year, decade, initial_key, initial_meter, initial_tempo_text."
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
        cols = ["record_id", "aria_name", "composer", "year_of_composition",
                "decade", "initial_key", "initial_meter", "initial_tempo_text"]
        out = _fmt_table(df, cols, limit=int(limit) if limit else 60)
        return f"{len(df)} match(es).\n{out}"


class DidoneMetadataTool(Tool):
    name = "get_aria_metadata"
    description = (
        "Get score-level metadata for one aria by record_id: composer, year, "
        "opera, character, opening key and key-change count, meter (and whether "
        "it changes), tempo marking and tempo family, da capo / repeat scheme, "
        "measure count, note density, lyric syllable density, and instrumentation. "
        "Does NOT contain the harmonic analysis -- use get_tonal_plan / get_harmony for that."
    )
    inputs = {"record_id": {"type": "string", "description": "Aria record_id, e.g. '0012'."}}
    output_type = "string"

    def forward(self, record_id: str) -> str:
        r = _row("metadata", record_id)
        if r is None:
            return f"No aria with record_id {record_id!r}."
        f = [
            f"record_id: {r['record_id']}",
            f"aria_name: {r['aria_name']}",
            f"composer: {r['composer']}",
            f"year_of_composition: {r['year_of_composition']}   decade: {r['decade']}",
            f"opera: {r['opera_name']}   character: {r['character'] or '(unknown)'}",
            f"initial_key: {r['initial_key'] or '(not parsed)'}   key_signature_changes: {r['n_key_signature_changes']}",
            f"initial_meter: {r['initial_meter']}   meter_family: {r['meter_family']}   meter_changes: {r['n_meter_changes']}   unique_meters: {r['unique_meters']}",
            f"initial_tempo_text: {r['initial_tempo_text']}   tempo_family: {r['tempo_family']}   tempo_changes: {r['n_tempo_changes']}",
            f"repeat_type: {r['repeat_type']}   has_segno: {r['has_segno']}   has_coda: {r['has_coda']}   has_fine: {r['has_fine']}",
            f"n_measures_written: {r['n_measures_written']}   vocal_measures: {r['vocal_measures_count']} (mm. {r['first_vocal_measure']}-{r['last_vocal_measure']})",
            f"note_density_per_measure: {r['note_density_per_measure']}   n_notes: {r['n_notes']}   n_rests: {r['n_rests']}   n_tuplets: {r['n_tuplets']}",
            f"lyric_density_per_measure: {r['lyric_density_per_measure']}   n_lyric_syllables: {r['n_lyric_syllables']}",
            f"harmony_annotation_density: {r['harmony_annotation_density']}   n_harmony_annotations: {r['n_harmony_annotations']}",
            f"instrumentation_signature: {r['instrumentation_signature']}",
        ]
        return "\n".join(f)


class DidoneTonalPlanTool(Tool):
    name = "get_tonal_plan"
    description = (
        "Get the tonal plan of one aria: the high-level sequence of key areas it "
        "visits (as a Roman-numeral / key-letter string), plus a per-segment "
        "breakdown -- for each tonal segment: measure span, region label, absolute "
        "key, opening and closing harmonic function, modulation type, and a "
        "confidence score. Use this for questions about modulation and overall "
        "tonal design."
    )
    inputs = {"record_id": {"type": "string", "description": "Aria record_id, e.g. '0012'."}}
    output_type = "string"

    def forward(self, record_id: str) -> str:
        ov = _row("tonal_plan_overview", record_id)
        seg = _load("tonal_plan_segments")
        seg = seg[seg["record_id"] == str(record_id).strip()]
        if ov is None and seg.empty:
            return f"No tonal plan for record_id {record_id!r}."
        head = f"tonal_plan: {ov['tonal_plan']}" if ov is not None else "tonal_plan: (overview row missing)"
        if seg.empty:
            return head + "\n(no per-segment breakdown)"
        cols = ["segment_index", "start_measure", "end_measure", "region_label",
                "absolute_key", "opening_function", "closing_function",
                "modulation_type", "parent_region", "confidence"]
        return f"{head}\n\nsegments:\n{_fmt_table(seg.sort_values('segment_index'), cols)}"


class DidoneHarmonyTool(Tool):
    name = "get_harmony"
    description = (
        "Get the chord-by-chord harmonic analysis of one aria over a measure "
        "range (Roman-numeral analysis). For each chord event: measure, beat, "
        "the chord label, its local function, the active tonal region/key, "
        "whether it is a secondary dominant (and of what), inversion, chord "
        "quality, and phrase-boundary flags. There is NO cadence label -- infer "
        f"cadences yourself from the progression and phrase_end flags. Max "
        f"{_MAX_HARMONY_SPAN} measures per call; page through a long aria with "
        "several calls."
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
            return (f"Range too wide ({b - a} measures). Ask for at most "
                    f"{_MAX_HARMONY_SPAN} measures at a time.")
        df = _load("harmony_events")
        df = df[(df["record_id"] == str(record_id).strip())
                & (pd.to_numeric(df["measure_number"], errors="coerce") >= a)
                & (pd.to_numeric(df["measure_number"], errors="coerce") <= b)]
        if df.empty:
            return f"No harmony events for record_id {record_id!r} in mm. {a}-{b}."
        cols = ["measure_number", "beat", "normalized_label", "local_function",
                "tonal_region", "is_secondary_dominant", "applied_to", "inversion",
                "quality_or_chord_type", "phrase_start", "phrase_end", "is_region_marker"]
        return f"mm. {a}-{b}, {len(df)} chord events:\n{_fmt_table(df, cols)}"


class DidoneSectionTonalPlanTool(Tool):
    name = "get_section_tonal_plan"
    description = (
        "Get the formal sections of one aria aligned with their local tonal "
        "plans: the written formal plan and the performed (post-repeat) plan, "
        "then per section -- symbol (R1, A1, B1, ...), section type "
        "(ritornello/text), measure span, the section's own tonal plan, and its "
        "dominant tonal region. Use this for questions about strophic/formal "
        "structure and where the tonal plan repeats (da capo detection)."
    )
    inputs = {"record_id": {"type": "string", "description": "Aria record_id, e.g. '0012'."}}
    output_type = "string"

    def forward(self, record_id: str) -> str:
        df = _load("section_tonal_plan")
        df = df[df["record_id"] == str(record_id).strip()]
        if df.empty:
            return f"No section tonal plan for record_id {record_id!r}."
        r0 = df.iloc[0]
        head = (f"written_plan: {r0['written_plan']}\n"
                f"performed_plan: {r0['performed_plan']}")
        cols = ["section_index", "section_symbol", "section_type",
                "section_start_measure", "section_end_measure",
                "section_tonal_plan", "dominant_tonal_region", "tonal_region_count"]
        return f"{head}\n\nsections:\n{_fmt_table(df.sort_values('section_index'), cols)}"


class DidoneTextStructureTool(Tool):
    name = "get_text_structure"
    description = (
        "Get the textual/strophic structure of one aria: the written formal "
        "plan, the repeat scheme (D.C./D.S./none), the performed plan, and the "
        "list of text sections with a preview of the sung words and their "
        "measure spans. Use this for questions about strophic form and text "
        "setting. Note: the actual full libretto is in get_aria_metadata's "
        "corpus only via vocal_text; this tool gives the section-level structure."
    )
    inputs = {"record_id": {"type": "string", "description": "Aria record_id, e.g. '0012'."}}
    output_type = "string"

    def forward(self, record_id: str) -> str:
        ov = _row("textual_plan_overview", record_id)
        sec = _load("textual_plan_sections")
        sec = sec[sec["record_id"] == str(record_id).strip()]
        if ov is None and sec.empty:
            return f"No textual structure for record_id {record_id!r}."
        head = (f"written_plan: {ov['written_plan']}\n"
                f"repeat_scheme: {ov['repeat_scheme']}\n"
                f"performed_plan: {ov['performed_plan']}\n"
                f"text_confidence_status: {ov['text_confidence_status']}") if ov is not None else "(overview row missing)"
        if sec.empty:
            return head + "\n(no section list)"
        cols = ["section_index", "symbol", "section_type", "start_measure",
                "end_measure", "text_preview"]
        return f"{head}\n\nsections:\n{_fmt_table(sec.sort_values('section_index'), cols)}"


def didone_tools() -> list[Tool]:
    """All six Didone query tools, freshly instantiated."""
    return [
        DidoneCorpusSearchTool(),
        DidoneMetadataTool(),
        DidoneTonalPlanTool(),
        DidoneHarmonyTool(),
        DidoneSectionTonalPlanTool(),
        DidoneTextStructureTool(),
    ]
