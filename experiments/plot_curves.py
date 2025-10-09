import csv, sys
import matplotlib.pyplot as plt

def read_csv(path):
    with open(path, newline="") as f:
        r=csv.DictReader(f)
        rows=[{k:float(v) if k!="epoch" else int(v) for k,v in row.items()} for row in r]
    return rows

def plot_xy(rows, xkey, ykeys, title, out_png):
    fig = plt.figure()
    for k in ykeys:
        plt.plot([r[xkey] for r in rows], [r[k] for r in rows], label=k)
    plt.title(title); plt.xlabel(xkey); plt.legend(); plt.tight_layout()
    fig.savefig(out_png, dpi=150); plt.close(fig)

def main():
    # genre
    try:
        g = read_csv("experiments/logs/genre_curve.csv")
        plot_xy(g, "epoch", ["acc","f1"], "Genre Baseline", "experiments/logs/genre_curve.png")
        print("Saved experiments/logs/genre_curve.png")
    except Exception as e:
        print("Skip genre:", e)
    # emotion
    try:
        e = read_csv("experiments/logs/emotion_curve.csv")
        plot_xy(e, "epoch", ["rmse_v","rmse_a"], "Emotion Baseline", "experiments/logs/emotion_curve.png")
        print("Saved experiments/logs/emotion_curve.png")
    except Exception as e:
        print("Skip emotion:", e)
    # mtl
    try:
        m = read_csv("experiments/logs/mtl_curve.csv")
        plot_xy(m, "epoch", ["acc","f1","rmse_v","rmse_a"], "MTL (Genre+Emotion)", "experiments/logs/mtl_curve.png")
        print("Saved experiments/logs/mtl_curve.png")
    except Exception as e:
        print("Skip mtl:", e)

if __name__ == "__main__":
    main()
