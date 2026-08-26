"""
Quickstart: run the full WeaveMuse manager agent in "only local" mode, on your own GPU.

Why this script exists: WeaveMuse's built-in `weavemuse gui` / `weavemuse terminal`
entry points ask you to pick a model tier (1: local, 2: HF-cloud hybrid, 3: all-remote)
interactively via stdin `input()`, which isn't convenient for a one-shot smoke test.
This script does the same "1. Only Local Models" setup non-interactively: the backbone
LLM loads locally via `TransformersModel` (model id + 4-bit quantization auto-picked
for your VRAM tier -- see weavemuse/utils/gpu_utils.py::detect_gpu_capabilities), and
tool_mode="hybrid" means NotaGenTool, StableAudioTool, ChatMusicianTool, and
AudioAnalysisTool (Qwen2-Audio) all run locally too, downloading weights on first use.

One exception: audio_analysis_agent always tries the remote AudioFlamingoTool (a
Gradio Space call) first regardless of mode, falling back to the local
AudioAnalysisTool only if that fails -- this codebase has no fully-offline
audio-analysis path. See scripts/quickstart_remote.py for the all-remote counterpart
of this script (no local GPU/weights needed at all, but billed against your HF
account's Inference Providers credits).

It also saves whatever the agent produced into an output directory and, for
music generations, pops up a small playback window (unless --no-play) so you
can listen to it on demand:
    - Any file paths mentioned in the agent's answer that exist on disk (e.g.
      from the notagen/stable_audio tools) are copied into the output dir.
    - Raw ABC notation in the answer (the backbone LLM sometimes composes ABC
      inline instead of calling the notagen tool) is saved as a .abc file and
      rendered to .mid with music21 -- this works even without MuseScore
      installed, unlike the notagen tool's own PDF/MIDI/MP3 conversion path.
    - Once generation/conversion finishes, a small Tk window lists whatever
      playable file(s) (.mid/.mp3/.wav) were produced with Play/Stop buttons --
      nothing plays until you click Play. On Windows this uses the OS's MCI
      player (no SoundFont/codec setup needed, and it actually supports Stop);
      elsewhere it falls back to opening the file with your default app.

Setup:
    1. Fill in HF_TOKEN in .env (needed for gated repos like
       stable-audio-open-1.0, and for the remote Audio Flamingo fallback call).
    2. uv sync --extra gpu --extra-index-url https://download.pytorch.org/whl/cu121
    3. Run:
        .venv/bin/python scripts/quickstart_local.py "Compose a short, cheerful piano piece in ABC notation."

Notes:
    - First run downloads model weights (backbone LLM + whichever tools you
      exercise) -- expect this to take a while and use several GB of disk.
    - Your VRAM tier determines which backbone model id gets used and whether
      it's 4-bit quantized; there's no override flag here, mirroring
      `weavemuse gui`'s own local-mode setup (WeaveMuseGUI.setup_local_model).
      Set WEAVEMUSE_FORCE_CPU=1 to force CPU mode instead.
    - max_steps is kept small; if the agent runs out of steps before finishing,
      rerun with a simpler prompt or raise MAX_STEPS below.
"""

import argparse
import datetime
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

MAX_STEPS = 4
DEFAULT_QUERY = "Compose a short, cheerful piano piece in ABC notation."
# Anchored to the repo root (this file's parent's parent), not the process's
# working directory -- PyCharm's default run configuration uses the script's
# own folder as cwd, which would otherwise silently create scripts/outputs/...
# instead of <repo_root>/outputs/... depending on how you launch this.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "quickstart_local"

# Absolute file paths (Windows drive-letter or POSIX-style) ending in a
# format WeaveMuse's tools produce.
_FILE_PATH_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|/)[^\s\"'`]+?\.(?:abc|xml|musicxml|pdf|mid|midi|mp3|wav)"
)
_CODE_FENCE_RE = re.compile(r"^```[a-zA-Z]*\n?|```$", re.MULTILINE)
PLAYABLE_EXTS = {".mid", ".midi", ".mp3", ".wav"}


def _extract_abc_block(text: str) -> str | None:
    """Find a standalone ABC tune (an "X:" header through a "K:" line and its
    tune-body) among the agent's answer. Splits on blank lines rather than
    using one hand-rolled regex, so trailing prose paragraphs in the same
    answer don't get swallowed into the tune body -- NotaGen/LLM-authored ABC
    formatting varies too much to trust a single lookahead for that boundary.
    """
    for paragraph in re.split(r"\n\s*\n", text):
        candidate = _CODE_FENCE_RE.sub("", paragraph).strip()
        if re.match(r"X:\s*\d+", candidate) and re.search(r"(?m)^K:", candidate):
            return candidate + "\n"
    return None


def _dedupe_shared_context_objects(score) -> None:
    """Work around a music21 ABC-parser bug: when an ABC source repeats an
    identical header line (e.g. the same "M:4/4" implicitly re-declared at
    each new line/system), the parser can reuse the *same* TimeSignature/
    KeySignature/Clef object instance across multiple measures instead of
    giving each measure its own copy. That later breaks stream.write('midi')
    with "the object (...) is already found in this Stream", even though the
    ABC itself is perfectly valid. Replace repeat instances with independent
    deep copies so each measure owns its own object.
    """
    import copy

    from music21 import clef, key, meter, stream

    seen_ids = set()
    for measure in score.recurse().getElementsByClass(stream.Measure):
        context_els = measure.getElementsByClass(
            (meter.TimeSignature, key.KeySignature, key.Key, clef.Clef)
        )
        for el in list(context_els):
            if id(el) in seen_ids:
                offset = el.offset
                measure.remove(el)
                measure.insert(offset, copy.deepcopy(el))
            else:
                seen_ids.add(id(el))


def _require_env(var_name: str) -> None:
    if not os.getenv(var_name):
        raise SystemExit(
            f"{var_name} is not set.\n"
            f"Fill in {var_name} in .env, or set it in your shell before running this script."
        )


def _check_vram_headroom(gpu_info) -> None:
    """Bail out early with a clear diagnosis if there isn't enough free VRAM
    for the 4-bit-quantized backbone model to load entirely on-GPU.

    Why this matters: if `device_map="auto"` can't fit the model in free
    VRAM, accelerate silently offloads the remainder to CPU RAM, and every
    forward pass then shuttles activations CPU<->GPU -- this turns ~7 tok/s
    (observed on an 8GB card with the model fully on-GPU) into roughly a
    minute or two *per token*, which looks like a hang rather than an error.
    Root cause seen in practice: a leftover/suspended process (e.g. a
    Ctrl-Z'd `weavemuse gui`) can sit on several GB of VRAM indefinitely
    without doing anything with it.
    """
    if not gpu_info.has_cuda:
        return
    needed_gb = 5.5  # ~ the 4-bit Qwen2.5-Coder-7B footprint incl. KV-cache headroom
    if gpu_info.free_vram_gb >= needed_gb:
        return

    print(
        f"\n⚠️  Only {gpu_info.free_vram_gb:.1f}GB VRAM free (~{needed_gb:.1f}GB needed for "
        f"{gpu_info.recommended_model_id} in 4-bit). Proceeding would likely force part of "
        f"the model onto CPU RAM, turning generation from ~7 tok/s into minutes per token."
    )
    try:
        procs = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        procs = ""
    if procs:
        print("Processes currently holding GPU memory:")
        print(procs)
        print("If one of these is stale (e.g. a suspended/Ctrl-Z'd run), free it with: kill -9 <pid>")
    raise SystemExit(
        "Not enough free VRAM to proceed safely -- see above. Free up VRAM and rerun, "
        "or set WEAVEMUSE_FORCE_CPU=1 to run on CPU deliberately (slow, but honest about it)."
    )


def _play_file_fallback(path: Path) -> None:
    """Last-resort playback via the OS's default file association (no
    play/stop control -- used when MCI isn't available, e.g. non-Windows, or
    fails to open a given file).
    """
    try:
        if sys.platform == "win32":
            os.startfile(str(path))
        elif sys.platform == "darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)
        print(f"Opened {path} with your system's default app.")
    except Exception as e:
        print(f"Could not open {path}: {e}\nOpen it manually to listen.")


class _MciPlayer:
    """Thin wrapper around the Windows MCI command interface (winmm, via
    ctypes -- no extra dependency), giving real play/stop control over a
    MIDI/audio file. MCI infers the device type from the file extension, so
    the same open/play/stop/close sequence works for .mid, .mp3, and .wav.

    This is what makes Stop possible, and it also sidesteps apps like VLC
    that can be set as the .mid file association but need a manually
    configured SoundFont before they'll make any sound at all.
    """

    def __init__(self) -> None:
        import ctypes

        self._winmm = ctypes.windll.winmm
        self._alias = "weavemuse_player"
        self._open_path: Path | None = None

    def _mci(self, command: str) -> int:
        import ctypes

        buf = ctypes.create_unicode_buffer(255)
        return self._winmm.mciSendStringW(command, buf, len(buf), None)

    def load(self, path: Path) -> bool:
        self.close()
        if self._mci(f'open "{path}" alias {self._alias}') != 0:
            return False
        self._open_path = path
        return True

    def play(self) -> bool:
        if self._open_path is None:
            return False
        self._mci(f"seek {self._alias} to start")  # so replay after Stop restarts
        return self._mci(f"play {self._alias}") == 0

    def stop(self) -> None:
        if self._open_path is not None:
            self._mci(f"stop {self._alias}")

    def close(self) -> None:
        if self._open_path is not None:
            self._mci(f"close {self._alias}")
            self._open_path = None


def show_player_gui(files: list[Path]) -> None:
    """Pop up a small window listing the generated playable file(s), with
    Play/Stop buttons. Nothing plays until the user clicks Play. Blocks until
    the window is closed.
    """
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title("WeaveMuse Player")
    root.resizable(False, False)

    frm = ttk.Frame(root, padding=16)
    frm.grid()

    ttk.Label(frm, text="Generated output:").grid(column=0, row=0, columnspan=2, sticky="w")

    names = [f.name for f in files]
    selected = tk.StringVar(value=names[0])
    dropdown = ttk.Combobox(
        frm, textvariable=selected, values=names, state="readonly", width=42
    )
    dropdown.grid(column=0, row=1, columnspan=2, pady=(4, 12), sticky="we")

    status_var = tk.StringVar(value="Ready")
    ttk.Label(frm, textvariable=status_var, foreground="#666").grid(
        column=0, row=3, columnspan=2, pady=(10, 0), sticky="w"
    )

    player = _MciPlayer() if sys.platform == "win32" else None

    def current_file() -> Path:
        return next(f for f in files if f.name == selected.get())

    def do_play() -> None:
        target = current_file()
        if player is not None and player.load(target) and player.play():
            status_var.set(f"Playing {target.name} ...")
        else:
            status_var.set(f"Opened {target.name} with your default app (no Stop control).")
            _play_file_fallback(target)

    def do_stop() -> None:
        if player is not None:
            player.stop()
        status_var.set("Stopped")

    dropdown.bind("<<ComboboxSelected>>", lambda _e: do_stop())

    btns = ttk.Frame(frm)
    btns.grid(column=0, row=2, columnspan=2)
    ttk.Button(btns, text="▶ Play", command=do_play, width=12).grid(column=0, row=0, padx=4)
    ttk.Button(btns, text="■ Stop", command=do_stop, width=12).grid(column=1, row=0, padx=4)

    def on_close() -> None:
        if player is not None:
            player.close()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    # Tk windows can otherwise open behind whatever has focus (terminal/IDE).
    root.lift()
    root.attributes("-topmost", True)
    root.after(200, lambda: root.attributes("-topmost", False))
    root.mainloop()


def save_and_play_result(result: str, output_dir: Path, play: bool) -> list[Path]:
    """Save the agent's result (and any generated music) into output_dir.

    Returns the list of files written/copied there.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    result_txt = output_dir / f"result_{timestamp}.txt"
    result_txt.write_text(result, encoding="utf-8")
    print(f"\nSaved raw agent result to {result_txt}")

    saved_files: list[Path] = []

    # Pick up any files the answer already points at on disk (e.g. from the
    # notagen/stable_audio tools' own output dirs).
    for match in _FILE_PATH_RE.finditer(result):
        src = Path(match.group(0))
        if src.is_file():
            dest = output_dir / src.name
            if src.resolve() != dest.resolve():
                shutil.copy2(src, dest)
            if dest not in saved_files:
                saved_files.append(dest)

    # If the answer contains raw ABC notation (the LLM composed it inline
    # instead of calling the notagen tool) and we didn't already pick up a
    # generated .abc file above, save it ourselves.
    abc_block = _extract_abc_block(result)
    if abc_block and not any(f.suffix == ".abc" for f in saved_files):
        abc_path = output_dir / f"generated_{timestamp}.abc"
        abc_path.write_text(abc_block, encoding="utf-8")
        saved_files.append(abc_path)
        print(f"Saved ABC notation to {abc_path}")

    # Render any ABC file to MIDI with music21, so it's listenable even
    # without a local MuseScore install (which the notagen tool's own
    # PDF/MIDI/MP3 conversion needs).
    for abc_file in [f for f in saved_files if f.suffix == ".abc"]:
        try:
            from music21 import converter

            score = converter.parse(str(abc_file), format="abc")
            _dedupe_shared_context_objects(score)
            midi_path = abc_file.with_suffix(".mid")
            score.write("midi", fp=str(midi_path))
            saved_files.append(midi_path)
            print(f"Rendered {abc_file.name} -> {midi_path.name} with music21")
        except Exception as e:
            print(f"Could not render {abc_file.name} to MIDI with music21: {e}")

    if not saved_files:
        print("No ABC notation or audio/MIDI files found in the result.")
        return saved_files

    print("\nOutput files:")
    for f in saved_files:
        print(f"  - {f}")

    if play:
        playable = [f for f in saved_files if f.suffix.lower() in PLAYABLE_EXTS]
        if playable:
            print(f"\nOpening player for: {', '.join(f.name for f in playable)}")
            show_player_gui(playable)
        else:
            print("\nNothing directly playable was produced (pass an ABC/audio-producing query, "
                  "or install MuseScore for the notagen tool's own MIDI/MP3 output).")

    return saved_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a one-shot WeaveMuse manager-agent query in only-local mode."
    )
    parser.add_argument(
        "query", nargs="*", help=f"Task for the agent (default: {DEFAULT_QUERY!r})."
    )
    parser.add_argument(
        "--output-dir", "-o", default=DEFAULT_OUTPUT_DIR,
        help=f"Directory to save generated files to (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--play", action=argparse.BooleanOptionalAction, default=True,
        help="Pop up a playback window for any generated ABC/audio (default: enabled). "
             "Pass --no-play to just save the files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    _require_env("HF_TOKEN")

    from smolagents import CodeAgent

    from weavemuse.agents.agents_as_tools import get_weavemuse_agents_and_tools
    from weavemuse.agents.models import TransformersModel
    from weavemuse.utils.gpu_utils import (
        check_force_cpu_mode,
        detect_gpu_capabilities,
        get_quantization_config,
        print_gpu_summary,
    )

    gpu_info = detect_gpu_capabilities(force_cpu=check_force_cpu_mode())
    print_gpu_summary(gpu_info)
    _check_vram_headroom(gpu_info)

    quantization_config = get_quantization_config(gpu_info)
    print("✅ Using 4-bit quantization for optimal VRAM usage" if quantization_config
          else "⚠️  Quantization disabled (CPU mode or not available)")

    print(f"Backbone model: {gpu_info.recommended_model_id} (local, device_map={gpu_info.device_map})")
    model = TransformersModel(
        model_id=gpu_info.recommended_model_id,
        trust_remote_code=True,
        device_map=gpu_info.device_map,
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        offload_buffers=True,
        quantization_config=quantization_config,
        # TransformersModel defaults to 4096 if unset; that's a needlessly high
        # worst-case per-step generation time for the short compositions this
        # script targets. 1536 comfortably covers a CodeAgent thought+code step
        # or a short ABC tune while capping how long a single runaway step can take.
        max_new_tokens=1536,
    )

    print("Setting up WeaveMuse agents and tools (tool_mode=hybrid)...")
    weavemuse_agents, weavemuse_tools = get_weavemuse_agents_and_tools(
        model=model,
        device_map=gpu_info.device_map,
        tool_mode="hybrid",
    )

    manager_agent = CodeAgent(
        tools=weavemuse_tools,
        model=model,
        managed_agents=weavemuse_agents,
        name="music_manager_agent",
        description=(
            "Manages music-related tasks, including web searches, advanced music analysis, "
            "symbolic music generation, audio music generation, and audio analysis using Audio Flamingo. "
            "When users upload audio files for analysis, forward the EXACT file path to the "
            "audio_flamingo_agent without checking if files exist first. "
            "If user asks to generate/compose music evaluate if symbolic (NotaGen) or audio (Stable Audio) is more appropriate. "
            "Use directly the corresponding tool. "
            "If a task is not related to music, ask the user to provide a music-related query. "
            "When you call generator tools like notagen or stable_audio (or any tool that returns file paths), "
            "do not keep working afterward. Call final_answer(...) immediately with a short summary and the returned paths. "
            "Do not plan further steps after final_answer."
        ),
        add_base_tools=True,
        stream_outputs=True,
        max_steps=MAX_STEPS,
        additional_authorized_imports=[],
    )

    query = " ".join(args.query) or DEFAULT_QUERY
    print(f"\nRunning query: {query!r}\n")
    result = manager_agent.run(query)

    print("\n=== RESULT ===")
    print(result)

    save_and_play_result(str(result), Path(args.output_dir), play=args.play)


if __name__ == "__main__":
    main()
