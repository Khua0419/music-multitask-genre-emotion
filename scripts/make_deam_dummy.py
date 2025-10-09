from pathlib import Path, PurePosixPath
import json

# 依赖之前跑过 scripts/make_tiny_dummy.py 生成的这两个文件
train_src = Path("data/lists/train_items.json")
val_src   = Path("data/lists/val_items.json")
if not train_src.exists() or not val_src.exists():
    raise SystemExit("请先运行: python scripts/make_tiny_dummy.py")

def load_fix(p):
    items = json.loads(p.read_text(encoding="utf-8"))
    out = []
    for it in items:
        out.append({
            "wav": str(PurePosixPath(it["wav"])),  # 统一路径分隔符
            "genre": 0,                            # 占位
            "emotion": it["emotion"]               # 使用已有的 V/A
        })
    return out

train = load_fix(train_src)
val   = load_fix(val_src)

Path("data/lists").mkdir(parents=True, exist_ok=True)
Path("data/lists/deam_train.json").write_text(json.dumps(train, indent=2), encoding="utf-8")
Path("data/lists/deam_val.json").write_text(json.dumps(val,   indent=2), encoding="utf-8")
print("Saved data/lists/deam_train.json & deam_val.json")
