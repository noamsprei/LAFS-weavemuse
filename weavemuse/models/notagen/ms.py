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

if platform.system() == "Linux":
    # MuseScore.AppImage only runs on Linux; on other platforms we fall back
    # to a system install below instead of trying to execute it.
    if not os.path.exists(apkname):
        download(
            filename=apkname,
            url="https://www.modelscope.cn/studio/Genius-Society/piano_trans/resolve/master/MuseScore.AppImage",
        )

    if not os.path.exists(extra_dir):
        subprocess.run(["chmod", "+x", f"./{apkname}"])
        subprocess.run([f"./{apkname}", "--appimage-extract"])

    file_dir = os.path.dirname(os.path.abspath(__file__))
    # only keep the part of the file_dir that is before the "/weavemuse/models/notagen/..."
    file_dir = file_dir[:file_dir.find("/weavemuse/models/")]
    MSCORE = os.path.join(file_dir, extra_dir, "AppRun")

    print("Running MuseScore from: ", MSCORE)
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
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