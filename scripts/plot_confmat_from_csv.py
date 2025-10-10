# scripts/plot_confmat_from_csv.py
import csv, sys
import numpy as np
import matplotlib.pyplot as plt
import itertools

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/plot_confmat_from_csv.py <confmat_csv> <out_png>")
        sys.exit(1)

    csv_path, out_png = sys.argv[1], sys.argv[2]

    with open(csv_path, "r", newline="") as f:
        r = csv.reader(f)
        rows = [row for row in r]

    # 第一行是列头：["", "pred_0", ..., "pred_9"]
    header = rows[0]
    body   = rows[1:]

    # 只取形如 ["true_0", n11, n12, ...] 的行
    data = []
    ytick = []
    for row in body:
        if not row: continue
        if not row[0].startswith("true_"): continue
        ytick.append(row[0].split("_",1)[1])
        # 将空白安全转为 0
        nums = []
        for x in row[1:]:
            try:
                x = (x or "").strip()
                nums.append(int(float(x)) if x!="" else 0)
            except:
                nums.append(0)
        data.append(nums)

    cm = np.array(data, dtype=np.int32)
    if cm.size == 0:
        print("[WARN] empty confmat parsed; check the CSV format.")
        sys.exit(2)

    xtick = [h.replace("pred_","") for h in header[1:]]
    plt.figure(figsize=(6,5), dpi=140)
    plt.imshow(cm, interpolation="nearest", cmap="viridis")
    plt.title("Confusion Matrix (Genre)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.colorbar()
    plt.xticks(np.arange(len(xtick)), xtick)
    plt.yticks(np.arange(len(ytick)), ytick)

    # 画数字
    vmax = cm.max() if cm.size else 1
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        val = cm[i, j]
        color = "white" if val > vmax * 0.6 else "black"
        plt.text(j, i, str(val), ha="center", va="center", color=color, fontsize=8)

    plt.tight_layout()
    plt.savefig(out_png)
    print("[OK] saved:", out_png)

if __name__ == "__main__":
    main()
