import os
import platform
import shutil
import requests
import subprocess
import time
from tqdm import tqdm

def download(filename, url):
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        chunk_size = 1024
        with open(filename, "wb") as file, tqdm(
            desc=f"Downloading {filename} from '{url}'...",
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)

    except Exception as e:
        print(f"Error: {e}, retrying...")
        time.sleep(10)
        download(filename, url)


apkname = "MuseScore.AppImage"
extra_dir = "squashfs-root"

MSCORE = None


def _looks_like_appimage(path: str) -> bool:
    """Cheap sanity check that `path` is actually an ELF/AppImage binary and
    not e.g. a truncated download or an error page/JSON blob served by a
    flaky host -- real MuseScore AppImages are hundreds of MB; anything
    under 10MB or missing the ELF magic bytes is not one. Without this
    check, a bad download (observed in practice: the host returning a small
    JSON error body instead of the binary) gets exec'd as-is and crashes
    with an opaque "Exec format error" instead of degrading gracefully.
    """
    try:
        if os.path.getsize(path) < 10 * 1024 * 1024:
            return False
        with open(path, "rb") as f:
            return f.read(4) == b"\x7fELF"
    except OSError:
        return False


if platform.system() == "Linux" and os.environ.get("MUSESCORE_PATH") and os.path.exists(os.environ["MUSESCORE_PATH"]):
    # Explicit override -- skip the AppImage download entirely.
    MSCORE = os.environ["MUSESCORE_PATH"]
    print("Running MuseScore from: ", MSCORE)
elif platform.system() == "Linux":
    # MuseScore.AppImage only runs on Linux; on other platforms we fall back
    # to a system install below instead of trying to execute it.
    if not os.path.isdir(extra_dir):
        if os.path.exists(apkname) and not _looks_like_appimage(apkname):
            # Stale/corrupt download left behind by a previous run -- remove
            # it so we don't keep trying to exec garbage.
            os.remove(apkname)
        if not os.path.exists(apkname):
            download(
                filename=apkname,
                url="https://www.modelscope.cn/studio/Genius-Society/piano_trans/resolve/master/MuseScore.AppImage",
            )
        if _looks_like_appimage(apkname):
            subprocess.run(["chmod", "+x", f"./{apkname}"])
            subprocess.run([f"./{apkname}", "--appimage-extract"])

    if os.path.isdir(extra_dir):
        file_dir = os.path.dirname(os.path.abspath(__file__))
        # only keep the part of the file_dir that is before the "/weavemuse/models/notagen/..."
        file_dir = file_dir[:file_dir.find("/weavemuse/models/")]
        MSCORE = os.path.join(file_dir, extra_dir, "AppRun")
        print("Running MuseScore from: ", MSCORE)
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    else:
        print(
            "⚠️  MuseScore.AppImage is unavailable (download failed or "
            "looked invalid -- the hosting service may be temporarily down). "
            "XML→PDF/MIDI/MP3 conversion will be skipped -- ABC and XML "
            "output still work. Install MuseScore yourself and set "
            "MUSESCORE_PATH to its executable to enable full conversion, or "
            "retry later."
        )
else:
    # Non-Linux (Windows/macOS): look for a MuseScore executable already on
    # this machine instead of the Linux-only AppImage. Set MUSESCORE_PATH to
    # point at yours if it isn't found automatically.
    _candidates = [
        os.environ.get("MUSESCORE_PATH"),
        shutil.which("MuseScore4"),
        shutil.which("MuseScore3"),
        shutil.which("mscore"),
        r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
        r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
        "/Applications/MuseScore 4.app/Contents/MacOS/mscore",
    ]
    MSCORE = next((c for c in _candidates if c and os.path.exists(c)), None)

    if MSCORE:
        print("Running MuseScore from: ", MSCORE)
    else:
        print(
            "\u26a0\ufe0f  MuseScore was not found on this system (the bundled "
            "MuseScore.AppImage only runs on Linux). XML\u2192PDF/MIDI/MP3 "
            "conversion will be skipped -- ABC and XML output still work. "
            "Install MuseScore and set the MUSESCORE_PATH environment "
            "variable to its executable to enable full conversion."
        )