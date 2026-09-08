# WeaveMuse sample bundle

Two short excerpts from the Diachronic corpus (IMSLP MIDI subset) plus two
example questions, so you have something concrete to run the agent on.

## What's here

| File | Source |
|---|---|
| `midi/josquin_mille-regretz_1549.mid` | Josquin Desprez, *Mille regretz* (NJE 28.25), c.1549 |
| `midi/chopin_nocturnes-op15_1830.mid`  | Chopin, *Nocturnes* Op.15, 1830 |
| `audio/*.wav` | First 30 s of each, rendered to 16 kHz mono |

Deliberately contrasting eras (Renaissance vocal polyphony vs Romantic piano),
so a style/period answer is either clearly right or clearly wrong.

**Caveat on the audio:** rendered with `pretty_midi.synthesize()`, i.e. additive
sine synthesis, not a sampled instrument (no fluidsynth/soundfont on the machine
these were made on). Pitch, rhythm, texture and voice count are faithful; timbre
is not. Good enough to smoke-test the pipeline and period/texture reasoning, not
for judging instrument-identification quality.

## The two questions

`questions.jsonl` — same schema as `data/eval/tasks_smoke.jsonl` on the
`eval-harness` branch (`task_id`, `query`, `category`, `tags`, `metadata`,
`attachments`). `attachments` holds file paths the agent is actually meant
to use (resolved relative to this file's own directory, and passed to the
manager agent via `agent.run(additional_args=...)`); `metadata` stays
agent-invisible eval bookkeeping (e.g. `ground_truth`) that must never reach
the model.

1. **s001 — audio analysis** (`audio_analysis_agent` / Audio Flamingo). The WAV
   path lives in `attachments.audio_file`; both the eval harness (via
   `additional_args`) and the Gradio UI (via manual upload) can feed it to the
   agent -- see both options below.
2. **s002 — symbolic generation** (`symbolic_music_agent` / NotaGen). Text-only,
   runs anywhere, exercises the ABC -> PDF/MIDI/MP3 render path.

Between them they cover both routes worth checking first: the one file-ingesting
path and the one generation path.

## Running them

### Option A -- Gradio UI (works on any machine, recommended first try)

    weavemuse gui

Both questions work here. For s001, use the **"Upload Audio File"** box to load
`audio/josquin_mille-regretz_1549.wav`, then paste the s001 `query` text into the
chat. Paste *only* the query string, not the whole JSON line -- the line contains
`ground_truth`, and handing that to the model defeats the point of asking.

### Option B -- eval harness (needs a CUDA GPU)

Two things to know before trying this:

1. **It lives on an unmerged branch.** A fresh clone gives you `main`, which has
   no `weavemuse/eval/` and no `scripts/run_eval.py`. You need:

        git checkout eval-harness

2. **It needs an NVIDIA GPU.** `run_eval.py` builds the backbone LLM locally via
   `TransformersModel` regardless of `--tool-mode`, so it wants real VRAM -- it
   won't run usefully on a laptop. It also requires `HF_TOKEN` to be set.

Given those, both tasks now run here (`--run-id` is required):

    export HF_TOKEN=...
    python scripts/run_eval.py \
      --dataset data/weavemuse_demo_data/questions.jsonl \
      --variants data/eval/variants_smoke.json \
      --run-id sample_check

s001's `attachments.audio_file` path is resolved to an absolute path at load time
and passed to the manager agent via `agent.run(additional_args=...)` -- it
arrives both as literal text in the task prompt and as a real `audio_file`
variable the manager's generated code can hand straight to
`audio_analysis_agent` (which must in turn forward it to the actual tool
call under that same `audio_file=` keyword -- see
`weavemuse/eval/README.md`'s note on why the key name matters). Inspect
`outputs/eval/sample_check/traces/s001/default.json`'s `"attachments"` field
and step trace to confirm the path actually reached the tool call.

## Verified

- `questions.jsonl` parses cleanly with the harness's own `load_tasks()`,
  including `attachments` path resolution.
- The UI's audio box is a `gr.Audio(type="filepath")` widget, so `.wav` is fine.
  (There's an unused legacy `upload_file()` in the same file whose whitelist is
  pdf/docx/txt -- ignore it, nothing calls it.)

## Known limitation: MIDI files aren't wired to any tool

The `midi/*.mid` files here are only the source material the `.wav`s were
rendered from -- no task references them as input, and there's currently no
way to make one influence NotaGen's output even if a task did:
`NotaGenTool`/`RemoteNotaGenTool` (`weavemuse/tools/notagen_tool.py`) take
only `period`/`composer`/`instrumentation` string labels and always generate
unconditionally from those -- there's no seed/reference-file argument at
all. Conditioning NotaGen on an existing MIDI file would need new capability
in that tool (and likely its underlying model), not just `attachments`
plumbing.
