import json, os, re

# Input lists
GTZAN_TRAIN = "data/lists/gtzan_train.json"
GTZAN_VAL   = "data/lists/gtzan_val.json"
DEAM_TRAIN  = "data/lists/deam_train.json"
DEAM_VAL    = "data/lists/deam_val.json"

OUT_DIR = "data/lists"
os.makedirs(OUT_DIR, exist_ok=True)

# Mapping for string labels in GTZAN
GENRE2ID = {
    'blues':0,'classical':1,'country':2,'disco':3,'hiphop':4,
    'jazz':5,'metal':6,'pop':7,'reggae':8,'rock':9
}

def first_key(d, candidates):
    for k in candidates:
        if k in d: return k
    return None

def get_audio_path(item):
    """Resolve an audio path; dataset expects the key 'wav' in the final JSON."""
    k = first_key(item, ["wav","path","filepath","file","audio","audio_path","relpath"])
    if k is None:
        raise KeyError(f"No audio path key in item keys={list(item.keys())[:10]}")
    return item[k]

def normalize_genre(val):
    """Return integer label in [0..9]."""
    if isinstance(val, str):
        v = val.strip().lower()
        if v in GENRE2ID: return GENRE2ID[v]
        if v.isdigit():
            iv = int(v); return iv-1 if 1 <= iv <= 10 else iv
        raise ValueError(f"Unknown genre string: {val}")
    iv = int(val)
    return iv-1 if 1 <= iv <= 10 else iv

def get_genre(item):
    """Extract genre from common keys and normalize."""
    k = first_key(item, ["genre","label","genre_id","y"])
    if k is None:
        # For DEAM we will fill a placeholder; return None here
        return None
    return normalize_genre(item[k])

def parse_emotion_value(e):
    """Parse emotion into [valence, arousal] with robust handling."""
    # list/tuple
    if isinstance(e, (list, tuple)) and len(e) >= 2:
        return [float(e[0]), float(e[1])]
    # dict with many possible keys
    if isinstance(e, dict):
        v_keys = ["val","valence","V","v"]
        a_keys = ["aro","arousal","A","a"]
        v = None; a = None
        for k in v_keys:
            if k in e: v = e[k]; break
        for k in a_keys:
            if k in e: a = e[k]; break
        if v is not None and a is not None:
            return [float(v), float(a)]
    # string: try json or regex like "[0.12, 0.34]"
    if isinstance(e, str):
        try:
            j = json.loads(e)
            return parse_emotion_value(j)
        except Exception:
            nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", e)
            if len(nums) >= 2:
                return [float(nums[0]), float(nums[1])]
    return None

def get_emotion(item):
    """Extract [val, arousal] from top-level or nested 'emotion' or val/aro keys."""
    # 1) direct 'emotion'
    if "emotion" in item:
        pv = parse_emotion_value(item["emotion"])
        if pv is not None: return pv
    # 2) separate keys
    k_v = first_key(item, ["val","valence","V","v"])
    k_a = first_key(item, ["aro","arousal","A","a"])
    if k_v is not None and k_a is not None:
        return [float(item[k_v]), float(item[k_a])]
    # Not found
    return None

def build_mtl_list(gtzan_list, deam_list, save_path):
    items = []

    # ---- GTZAN -> classification items (with dummy emotion) ----
    for it in gtzan_list:
        try:
            ap = get_audio_path(it)
            g  = get_genre(it)
            if g is None:
                raise KeyError("Missing genre for GTZAN item")
            items.append({
                "task": "genre",
                "wav": ap,
                "genre": g,
                "emotion": [0.0, 0.0]   # placeholder to satisfy dataset keys
            })
        except Exception as e:
            print(f"[WARN] skip GTZAN item: {e}")

    # ---- DEAM -> regression items (with dummy genre) ----
    for it in deam_list:
        try:
            ap = get_audio_path(it)
            va = get_emotion(it)
            if va is None:
                raise KeyError(f"No val/arousal info in item keys={list(it.keys())[:10]}")
            items.append({
                "task": "emotion",
                "wav": ap,
                "genre": -1,           # placeholder to satisfy dataset keys
                "emotion": va
            })
        except Exception as e:
            print(f"[WARN] skip DEAM item: {e}")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved {len(items)} items -> {save_path}")

def load_json(p):
    with open(p,"r",encoding="utf-8") as f:
        return json.load(f)

# ---- Main ----
gtr = load_json(GTZAN_TRAIN)
gtv = load_json(GTZAN_VAL)
dtr = load_json(DEAM_TRAIN)
dv  = load_json(DEAM_VAL)

print("GTZAN train first keys:", list(gtr[0].keys()) if gtr else [])
print("DEAM  train first keys:", list(dtr[0].keys()) if dtr else [])

build_mtl_list(gtr, dtr, os.path.join(OUT_DIR, "train_items.json"))
build_mtl_list(gtv, dv,  os.path.join(OUT_DIR, "val_items.json"))
