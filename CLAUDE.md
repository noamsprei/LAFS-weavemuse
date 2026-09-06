# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## How to respond

Answer one step at a time. Do one thing, show the result, stop, and wait. Do not
chain multiple steps or pre-empt the next question. Be concise: no preamble, no
recap, no unsolicited alternatives or "you could also" lists. If a question is
simple, answer it in a sentence or two. Expand only when asked.

## What this is

WeaveMuse is a multi-agent music AI framework built on `smolagents` (Hugging Face's agent
library). A manager `CodeAgent` orchestrates specialized sub-agents/tools for symbolic music
generation (NotaGen, ABC notation), audio generation (Stable Audio Open), music understanding
(ChatMusician), and audio analysis (Audio Flamingo / Qwen2-Audio), plus web search. It ships
both a Gradio web UI and a terminal UI, and is GPU-optimized but degrades to CPU/remote-only
modes when VRAM is limited.

## Setup and running

Package manager is `uv` (preferred) with a checked-in `uv.lock`; `requirements.txt` and
`pyproject.toml` also exist but `uv.lock` is the source of truth for reproducible installs.

```bash
# Install (GPU/CUDA 12.1)
uv sync --extra-index-url https://download.pytorch.org/whl/cu121

# Full dev install (all optional extras)
uv sync --extra dev --extra gpu --extra remote --extra audio --extra music --extra-index-url https://download.pytorch.org/whl/cu121

# Run
weavemuse gui         # Gradio web interface (default), served via app.py / weavemuse/interfaces/gui.py
weavemuse terminal     # terminal interface, weavemuse/interfaces/terminal_interface.py
python app.py          # alternate entrypoint used for `gradio` hot-reload dev (gr.NO_RELOAD guard)
```

`weavemuse` CLI entry point is `weavemuse.cli:main` (registered in `pyproject.toml`
`[project.scripts]`). There is no test suite, linter config invocation, or CI in this repo
currently — `pyproject.toml` declares `pytest`/`black`/`isort`/`flake8`/`mypy` as dev extras but
no tests or lint configs exist yet, so don't assume standard `pytest`/`make lint` commands work.

GPU mode requires an NVIDIA GPU with CUDA 12.1+; set `WEAVEMUSE_FORCE_CPU=1` to force CPU mode.
Model/runtime config comes from environment variables (`HF_TOKEN`, `DEVICE`, `TORCH_DTYPE`,
`HOST`/`PORT`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.) loaded via
`weavemuse/utils/config.py::MusicAgentConfig`.

## Architecture

### Agent hierarchy

A single **manager agent** (`smolagents.CodeAgent`, built in `app.py` /
`weavemuse/interfaces/gui.py`) is given the backbone LLM plus `managed_agents` — each a
specialized `CodeAgent` built by `weavemuse/agents/agents_as_tools.py::get_weavemuse_agents_and_tools()`:

- `symbolic_music_agent` — wraps `NotaGenTool`/`RemoteNotaGenTool`; generates ABC notation →
  PDF/MIDI/MusicXML/MP3.
- `audio_analysis_agent` — wraps `AudioFlamingoTool` (remote, tried first) and, when not
  remote-only, `AudioAnalysisTool` (local Qwen2-Audio) as a fallback.
- `audio_generation_agent` — wraps `StableAudioTool`/`RemoteStableAudioTool`.
- `web_search_agent` — wraps `smolagents.WebSearchTool`.

`ChatMusicianTool` is passed directly to the manager as a top-level tool (not wrapped in its own
sub-agent) unless running in remote-only mode.

The backbone LLM (`TransformersModel` in `weavemuse/agents/models.py`, a local
`smolagents.Model` subclass around HF `transformers`, or `smolagents.InferenceClientModel` for
remote) is auto-selected by GPU tier — see below. All agent intelligence (task routing, tool
selection, response quality) is bottlenecked by this backbone model's capability.

### Local vs. remote tool duality

Every heavy model-backed capability has two implementations selected by a `tool_mode` /
`remote_only` flag threaded through `get_weavemuse_agents_and_tools()`:

- **Local**: subclasses `LazyLoadableTool` → `ManagedTransformersTool` or `ManagedDiffusersTool`
  (`weavemuse/tools/base_tools.py`). Registers itself with the global `vram_manager`
  (`weavemuse/tools/memory_manager.py`) for lazy load, LRU eviction across a max of 2
  simultaneously loaded tools, and VRAM-budget-aware unloading. `forward()` bridges smolagents'
  sync interface to this manager's async load/call/evict machinery.
- **Remote**: plain `smolagents.Tool` subclasses prefixed `Remote*` (e.g. `RemoteNotaGenTool`,
  `RemoteStableAudioTool`) or backed by a Gradio Client / HF Inference API call (e.g.
  `AudioFlamingoTool`), requiring no local GPU/VRAM.

When adding a new model-backed tool, follow this pattern: a `Remote*Tool(Tool)` for
zero-resource use, and a local variant extending `ManagedTransformersTool` /
`ManagedDiffusersTool` implementing `_load_model` / `_call_model` (and optionally `_unload_model`).

### GPU/VRAM auto-configuration

`weavemuse/utils/gpu_utils.py::detect_gpu_capabilities()` probes CUDA availability and free VRAM,
buckets into a `VRAMTier` (LOW <20GB / MEDIUM <45GB / HIGH ≥45GB effective VRAM), and returns a
`GPUInfo` with a tier-appropriate backbone model id and whether to 4-bit-quantize. `app.py` and
`weavemuse/interfaces/gui.py` call this at startup to configure the manager's backbone model
before agents are built. Respect `WEAVEMUSE_FORCE_CPU` when touching this path.

### Interfaces

`weavemuse/interfaces/gradio_interface.py::WeaveMuseInterface` wraps a built `manager_agent` for
the Gradio chat UI (file upload, score/audio rendering); `weavemuse/interfaces/gui.py` is the
higher-level class the CLI launches, which interactively prompts the user for a model
tier/mode (local / HF-cloud / all-remote) before constructing agents —
`weavemuse/interfaces/terminal_interface.py` is the terminal-only equivalent. `app.py` is a
third, more direct entrypoint (mainly for `gradio` dev hot-reload) that builds the manager agent
inline rather than going through `WeaveMuseGUI`.

### NotaGen submodule

`weavemuse/models/notagen/` is a fairly self-contained symbolic-music-generation subsystem
(inference, ABC↔MusicXML conversion via `abc2xml.py`/`convert.py`, quantization) consumed by
`weavemuse/tools/notagen_tool.py`. Treat it as a semi-independent unit when tracing NotaGen bugs.
