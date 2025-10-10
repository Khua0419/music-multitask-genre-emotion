# scripts/plot_confmat_from_csv.py
# Robust CSV -> PNG (no GUI needed)

import os, csv, itertools
import numpy as np

import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

CSV_IN  = r"experiments/logs/genre_confmat.csv"
PNG_OUT = r"experiments/logs/genre_confmat.png"

def read_cm(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rdr = csv.reader(f)
        for r in rdr:
            if not r or all(c.strip()=="" for c in r):
                continue
            rows.append([c.strip() for c in r])

    if not rows:
        raise RuntimeError("Empty CSV: " + csv_path)

    header = rows[0]               # ["", "pred_0", "pred_1", ...]
    data_rows = rows[1:]           # ["true_0", n, n, ...], ...

    cm = []
    for r in data_rows:
        # r[0] like "true_0" -> ignore; then numeric cells
        nums = []
        for c in r[1:]:
            if c == "": c = "0"
            # allow int or float
            nums.append(int(float(c)))
        cm.append(nums)

    cm = np.array(cm, dtype=int)
    labels = list(range(cm.shape[0]))
    return cm, labels

def plot_cm(cm, labels, out_png):
    fig = plt.figure(figsize=(6,5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Confusion Matrix (Genre)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.colorbar()
    plt.xticks(range(len(labels)), labels)
    plt.yticks(range(len(labels)), labels)

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=7)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=180)
    print("Saved:", out_png)

if __name__ == "__main__":
    cm, labels = read_cm(CSV_IN)
    plot_cm(cm, labels, PNG_OUT)
