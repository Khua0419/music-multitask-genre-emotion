# scripts/plot_learning_curve.py
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path = r"experiments/logs/genre_curve.csv"
png_path = r"experiments/logs/genre_learning_curves.png"

epochs, loss, acc, f1 = [], [], [], []
with open(csv_path, "r", encoding="utf-8") as f:
    rdr = csv.DictReader(f)
    for r in rdr:
        epochs.append(int(r["epoch"]))
        # 兼容可能的字段名
        loss.append(float(r.get("val_loss") or r.get("loss") or 0.0))
        acc.append(float(r.get("acc", 0.0)))
        f1.append(float(r.get("f1", 0.0)))

plt.figure(figsize=(8,4))
plt.plot(epochs, loss, label="val_loss")
plt.plot(epochs, acc,  label="val_acc")
plt.plot(epochs, f1,   label="val_f1")
plt.xlabel("epoch"); plt.ylabel("score/loss"); plt.title("Genre learning curves")
plt.legend(); plt.tight_layout()
os.makedirs(os.path.dirname(png_path), exist_ok=True)
plt.savefig(png_path, dpi=160)
print("Saved:", png_path)
