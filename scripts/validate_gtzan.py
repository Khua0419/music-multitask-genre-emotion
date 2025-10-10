# scripts/validate_gtzan.py
# Find unreadable/corrupt wavs under data\GTZAN_raw and move them to data\GTZAN_bad

import os, glob, shutil
import soundfile as sf
import librosa

ROOT = r"data\GTZAN_raw"
BAD_DIR = r"data\GTZAN_bad"

os.makedirs(BAD_DIR, exist_ok=True)

bad = []
checked = 0

for p in glob.glob(os.path.join(ROOT, "*", "*.wav")):
    checked += 1
    try:
        # 1) try soundfile
        x, sr = sf.read(p, dtype="float32", always_2d=False)
    except Exception:
        try:
            # 2) fallback: librosa decode at native sr (no resample)
            y, fs = librosa.load(p, sr=None, mono=True)
        except Exception:
            bad.append(p)

print(f"Checked: {checked}, bad: {len(bad)}")
for p in bad:
    cls = os.path.basename(os.path.dirname(p))
    dst = os.path.join(BAD_DIR, f"{cls}__{os.path.basename(p)}")
    try:
        shutil.move(p, dst)
        print("Moved:", p, "->", dst)
    except Exception as e:
        print("Fail move:", p, e)

if bad:
    print("\nBad files were moved to", BAD_DIR)
    print("Now re-generate item lists and re-run training.")
else:
    print("\nNo bad files found.")
