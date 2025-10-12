# scripts/make_full_splits_balanced.py
# Create balanced (stratified) train/val splits for GTZAN:
# each genre -> 80 train, 20 val (≈100 files/genre)

import os, json, random, glob
from pathlib import Path

random.seed(0)

ROOT = Path(".")
RAW  = ROOT / "data" / "GTZAN_raw"
OUTD = ROOT / "data" / "lists"
OUTD.mkdir(parents=True, exist_ok=True)

# genre -> id (fixed order)
genre2id = {
    "blues":0, "classical":1, "country":2, "disco":3, "hiphop":4,
    "jazz":5, "metal":6, "pop":7, "reggae":8, "rock":9,
}

def is_good_wav(p):
    # quick sanity check using soundfile; skip files soundfile can't read
    try:
        import soundfile as sf
        _x, _sr = sf.read(str(p), frames=1024, dtype="float32")
        return True
    except Exception:
        return False

items_by_genre = {g: [] for g in genre2id}
for g in genre2id:
    gdir = RAW / g
    wavs = sorted(glob.glob(str(gdir / "*.wav")))
    for w in wavs:
        if is_good_wav(w):
            items_by_genre[g].append({"wav": w.replace("\\", "/"),
                                      "genre": genre2id[g],
                                      "emotion": [0.0, 0.0]})  # placeholder

# build balanced splits
train, val = [], []
for g, lst in items_by_genre.items():
    random.shuffle(lst)
    # Typically 100 songs per category; to be safe, use 80 and the remainder.
    n_tr = min(80, max(0, len(lst)-20))
    train.extend(lst[:n_tr])
    val.extend(lst[n_tr:])

# shuffle (optional, only for diversity)
random.shuffle(train)
random.shuffle(val)

# save
train_out = OUTD / "gtzan_train.json"
val_out   = OUTD / "gtzan_val.json"
with open(train_out, "w", encoding="utf-8") as f:
    json.dump(train, f, ensure_ascii=False, indent=2)
with open(val_out, "w", encoding="utf-8") as f:
    json.dump(val, f, ensure_ascii=False, indent=2)

# small report
from collections import Counter
print("saved:", train_out, ",", val_out)
print("len train/val:", len(train), len(val))
print("train per-genre:", Counter([x["genre"] for x in train]))
print("val   per-genre:", Counter([x["genre"] for x in val]))
