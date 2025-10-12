# scripts/gtzan_to_items.py
from pathlib import Path
import json

ROOT = Path("data/GTZAN_raw")  # Expected Directory Structure: data/GTZAN_raw/<genre>/<wav>
if not ROOT.exists():
    raise SystemExit("目录 data/GTZAN_raw 不存在，请先把GTZAN解压到这个路径。")

genres = sorted([p.name for p in ROOT.iterdir() if p.is_dir()])
label_map = {g: i for i, g in enumerate(genres)}
items = []
for g in genres:
    for wav in (ROOT / g).rglob("*.wav"):
        items.append({
            "wav": str(wav.as_posix()),
            "genre": label_map[g],
            "emotion": [0.0, 0.0]  # Placeholder, not needed for single-task use
        })

# 8:2 Partitioning (Simple sequential partitioning, later changeable to hierarchical partitioning by filename)
n = int(0.8 * len(items))
Path("data/lists").mkdir(parents=True, exist_ok=True)
(Path("data/lists/gtzan_train.json")
 ).write_text(json.dumps(items[:n], indent=2), encoding="utf-8")
(Path("data/lists/gtzan_val.json")
 ).write_text(json.dumps(items[n:], indent=2), encoding="utf-8")
print(f"Saved: data/lists/gtzan_train.json ({n}) , gtzan_val.json ({len(items)-n})")
