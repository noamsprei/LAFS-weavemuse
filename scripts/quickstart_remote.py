"""
Quickstart: run the full WeaveMuse manager agent in "all remote" mode.

Why this script exists: WeaveMuse's built-in `weavemuse gui` / `weavemuse terminal`
entry points ask you to pick a model tier (1: local, 2: HF-cloud hybrid, 3: all-remote)
interactively via stdin `input()`, which isn't convenient for a one-shot smoke test.
This script does the same "all remote" setup non-interactively (remote_only=True:
RemoteNotaGenTool, RemoteStableAudioTool, AudioFlamingoTool, web search -- no local
model weights downloaded or loaded) with WeaveMuse's default backbone: Qwen3-Coder-30B
via Hugging Face's Inference Providers ("featherless-ai" provider), billed against your
HF account's Inference Providers credits (HF grants some free credits monthly).

It also saves whatever the agent produced into an output directory and, for
music generations, pops up a small playback window (unless --no-play) so you
can listen to it on demand:
    - Any file paths mentioned in the agent's answer that exist on disk (e.g.
      from the notagen/stable_audio remote tools) are copied into the output dir.
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
    1. Copy .env.example to .env and fill in HF_TOKEN.
    2. uv sync --extra remote   (installs anthropic/openai/google clients if not already present)
    3. Run:
        .venv\\Scripts\\python.exe scripts\\quickstart_remote.py "Compose a short, cheerful piano piece in ABC notation."

Notes:
    - The remote tools call third-party Hugging Face Spaces (cold starts can take
      a minute or two), and the backbone call goes through HF Inference Providers --
      both draw on your Hugging Face account.
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
BACKBONE_MODEL_ID = "Qwen/Qwen3-Coder-30B-A3B-Instruct"
# NOTE: WeaveMuse's own gui.py hardcodes provider="nebius", but as of this
# writing HF's inferenceProviderMapping for this model only lists
# featherless-ai and scaleway as live -- nebius 404s. Verify at
# https://huggingface.co/api/models/{model}?expand[]=inferenceProviderMapping
# if this starts failing again.
BACKBONE_PROVIDER = "featherless-ai"
DEFAULT_QUERY = "Compose a short, cheerful piano piece in ABC notation."
# Anchored to the repo root (this file's parent's parent), not the process's
# working directory -- PyCharm's default run configuration uses the script's
# own folder as cwd, which would otherwise silently create scripts/outputs/...
# instead of <repo_root>/outputs/... depending on how you launch this.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "quickstart_remote"

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
            f"Copy .env.example to .env and fill in {var_name}, "
            f"or set it in your shell before running this script."
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
    # notagen/stable_audio remote tools' own output dirs).
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
        description="Run a one-shot WeaveMuse manager-agent query in all-remote mode."
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
    os.environ.setdefault("WEAVEMUSE_FORCE_CPU", "1")

    from smolagents import CodeAgent, InferenceClientModel

    from weavemuse.agents.agents_as_tools import get_weavemuse_agents_and_tools

    print(f"Backbone model: {BACKBONE_MODEL_ID} (via HF Inference Providers / {BACKBONE_PROVIDER})")
    model = InferenceClientModel(model_id=BACKBONE_MODEL_ID, provider=BACKBONE_PROVIDER)

    print("Setting up WeaveMuse agents and tools (tool_mode=remote)...")
    weavemuse_agents, weavemuse_tools = get_weavemuse_agents_and_tools(
        model=model,
        device_map="cpu",
        tool_mode="remote",
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
            "If a task is not related to music, ask the user to provide a music-related query."
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
