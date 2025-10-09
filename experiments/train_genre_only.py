import csv, os, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from src.datasets.audio_dataset import AudioMultiTaskSet
from src.models.multitask import MultiTaskNet

def plot_confmat(y_true, y_pred, out_png):
    cm = confusion_matrix(y_true, y_pred, labels=sorted(set(y_true)))
    fig = plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title("Confusion Matrix (Genre)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.colorbar()
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def main():
    cfg={"train_items":"data/lists/gtzan_train.json",
         "val_items":"data/lists/gtzan_val.json",
         "num_genres":10, "lr":1e-3, "bs":2, "epochs":5}
    os.makedirs("experiments/logs", exist_ok=True)
    os.makedirs("experiments/checkpoints", exist_ok=True)
    log_csv = "experiments/logs/genre_curve.csv"
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tr=AudioMultiTaskSet(cfg["train_items"]); va=AudioMultiTaskSet(cfg["val_items"])
    net=MultiTaskNet(num_genres=cfg["num_genres"]).to(dev)
    opt=torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    dl_tr=DataLoader(tr,batch_size=cfg["bs"],shuffle=True)
    dl_va=DataLoader(va,batch_size=cfg["bs"])
    with open(log_csv, "w", newline="") as f:
        w=csv.writer(f); w.writerow(["epoch","acc","f1"])
        for ep in range(cfg["epochs"]):
            net.train()
            for b in dl_tr:
                x=b["spec"].to(dev); g=b["genre"].to(dev)
                lg,_=net(x)
                loss=torch.nn.functional.cross_entropy(lg,g)
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval(); Gt=[]; Gp=[]
            with torch.no_grad():
                for b in dl_va:
                    x=b["spec"].to(dev); g=b["genre"].numpy()
                    lg,_=net(x); Gt.extend(g); Gp.extend(lg.argmax(1).cpu().numpy())
            acc=accuracy_score(Gt,Gp); f1=f1_score(Gt,Gp,average="macro",zero_division=0)
            print(f"[Genre|Epoch {ep+1}] acc={acc:.3f} f1={f1:.3f}")
            w.writerow([ep+1, f"{acc:.6f}", f"{f1:.6f}"])
            f.flush()
    # 保存模型 & 混淆矩阵
    torch.save(net.state_dict(), "experiments/checkpoints/genre_last.pt")
    plot_confmat(Gt, Gp, "experiments/logs/genre_confmat.png")

if __name__=="__main__":
    main()
