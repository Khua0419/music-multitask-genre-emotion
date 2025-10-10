# scripts/plot_learning_curve.py
import csv, sys, math
import matplotlib.pyplot as plt

if len(sys.argv) < 3:
    print("Usage: python scripts/plot_learning_curve.py <curve.csv> <out.png>")
    sys.exit(1)

csv_path, out_path = sys.argv[1], sys.argv[2]

def parse(x):
    try:
        if x is None:
            return None
        x = str(x).strip()
        if x == "":
            return None
        v = float(x)
        if math.isfinite(v):
            return v
    except Exception:
        return None
    return None

epochs, val_loss, val_acc, val_f1 = [], [], [], []

with open(csv_path, "r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        e  = parse(row.get("epoch"))
        vl = parse(row.get("val_loss") or row.get("loss"))
        va = parse(row.get("val_acc")  or row.get("acc"))
        vf = parse(row.get("val_f1")   or row.get("f1"))
        if e is None:
            continue
        epochs.append(e)
        val_loss.append(vl if vl is not None else float("nan"))
        val_acc.append(va if va is not None else float("nan"))
        val_f1.append(vf if vf is not None else float("nan"))

print(f"[INFO] points: {len(epochs)} from {csv_path}")

plt.figure(figsize=(12,6))
if any(math.isfinite(x) for x in val_loss):
    plt.plot(epochs, val_loss, label="val_loss")
if any(math.isfinite(x) for x in val_acc):
    plt.plot(epochs, val_acc,  label="val_acc")
if any(math.isfinite(x) for x in val_f1):
    plt.plot(epochs, val_f1,   label="val_f1")

plt.title("Genre Baseline")
plt.xlabel("epoch"); plt.ylabel("score/loss")
plt.grid(True, alpha=0.2); plt.legend()
plt.tight_layout()
plt.savefig(out_path, dpi=160)
print(f"[OK] saved: {out_path}")
