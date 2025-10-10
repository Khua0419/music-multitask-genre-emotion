# scripts/plot_learning_curve.py
import csv, sys
import matplotlib.pyplot as plt

CANDIDATE_ACC = ["acc", "val_acc", "val/acc", "valid_acc", "accuracy"]
CANDIDATE_F1  = ["f1", "val_f1", "val/f1", "macro_f1"]

def safe_float(x):
    try:
        if x is None: return None
        s = str(x).strip()
        if s == "": return None
        return float(s)
    except Exception:
        return None

def pick_col(row, keys):
    for k in keys:
        if k in row:
            v = safe_float(row.get(k))
            if v is not None:
                return v, k
    return None, None

def main():
    if len(sys.argv) < 3:
        print("Usage: python scripts/plot_learning_curve.py <curve_csv> <out_png>")
        sys.exit(1)

    csv_path, out_png = sys.argv[1], sys.argv[2]

    epochs, accs, f1s = [], [], []
    acc_key_used, f1_key_used = None, None

    with open(csv_path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            ep = safe_float(row.get("epoch"))
            acc, kacc = pick_col(row, CANDIDATE_ACC)
            f1,  kf1  = pick_col(row, CANDIDATE_F1)
            if ep is None or acc is None or f1 is None:
                continue
            epochs.append(ep); accs.append(acc); f1s.append(f1)
            acc_key_used = acc_key_used or kacc
            f1_key_used  = f1_key_used  or kf1

    print(f"[INFO] read points: {len(epochs)} from {csv_path}")
    if len(epochs) == 0:
        print("[HINT] Header columns found:", getattr(r, "fieldnames", None))
        print("[HINT] I look for epoch + one of", CANDIDATE_ACC, "and one of", CANDIDATE_F1)
        sys.exit(2)

    plt.figure(figsize=(6,4), dpi=140)
    plt.plot(epochs, accs, label=f"{acc_key_used}", marker="o", linewidth=2)
    plt.plot(epochs, f1s,  label=f"{f1_key_used}",  marker="o", linewidth=2)
    plt.xlabel("epoch"); plt.ylabel("score"); plt.title("Genre Baseline")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()
    plt.savefig(out_png)
    print("[OK] saved:", out_png, "| keys:", acc_key_used, f1_key_used)

if __name__ == "__main__":
    main()

