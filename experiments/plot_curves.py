import os
import csv
import numpy as np
import matplotlib.pyplot as plt

LOG_DIR = "experiments/logs"
os.makedirs(LOG_DIR, exist_ok=True)

def read_csv(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            # 把能转成数字的字段转成 float，其他保持原样
            conv = {}
            for k, v in r.items():
                try:
                    conv[k] = float(v)
                except Exception:
                    conv[k] = v
            rows.append(conv)
    # 简单的“迷你 DataFrame”实现（避免依赖 pandas）
    class DF(list):
        @property
        def columns(self):
            return rows[0].keys() if rows else []
        def __getitem__(self, key):
            if isinstance(key, str):
                return [row.get(key, None) for row in rows]
            return super().__getitem__(key)
    return DF(rows)

def plot_xy(df, x, ys, title, out_png):
    plt.figure()
    xs = df[x]
    for y in ys:
        try:
            ys_ = df[y]
            plt.plot(xs, ys_, marker='o', linewidth=2, label=y)
        except Exception:
            pass
    plt.title(title)
    plt.xlabel(x)
    plt.ylabel("score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

def main():
    # 1) Genre 曲线
    try:
        g = read_csv(os.path.join(LOG_DIR, "genre_curve.csv"))
        plot_xy(g, "epoch", ["acc", "f1"], "Genre Baseline",
                os.path.join(LOG_DIR, "genre_curve.png"))
        print("Saved", os.path.join(LOG_DIR, "genre_curve.png"))
    except Exception as e:
        print("Skip genre curve:", e)

    # 2) Genre 混淆矩阵
    try:
        cm_path = os.path.join(LOG_DIR, "genre_confmat.csv")
        if os.path.exists(cm_path):
            mat = []
            with open(cm_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    mat.append([float(x) for x in row])
            mat = np.array(mat)
            plt.figure()
            plt.imshow(mat, cmap="viridis")
            plt.title("Confusion Matrix (Genre)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar()
            plt.tight_layout()
            out_png = os.path.join(LOG_DIR, "genre_confmat.png")
            plt.savefig(out_png, dpi=180)
            plt.close()
            print("Saved", out_png)
    except Exception as e:
        print("Skip genre confmat:", e)

    # 3) Emotion 曲线
    try:
        e = read_csv(os.path.join(LOG_DIR, "emotion_curve.csv"))
        plot_xy(e, "epoch", ["rmse_v", "rmse_a"], "Emotion Baseline",
                os.path.join(LOG_DIR, "emotion_curve.png"))
        print("Saved", os.path.join(LOG_DIR, "emotion_curve.png"))
    except Exception as e:
        print("Skip emotion curve:", e)

    # 4) MTL 综合曲线
    try:
        m = read_csv(os.path.join(LOG_DIR, "mtl_curve.csv"))
        plot_xy(m, "epoch", ["acc", "f1", "rmse_v", "rmse_a"], "MTL (Genre+Emotion)",
                os.path.join(LOG_DIR, "mtl_curve.png"))
        print("Saved", os.path.join(LOG_DIR, "mtl_curve.png"))
    except Exception as e:
        print("Skip mtl curve:", e)

if __name__ == "__main__":
    main()
