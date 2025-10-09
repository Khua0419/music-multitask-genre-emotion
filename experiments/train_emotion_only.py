import csv, os, numpy as np, torch
from torch.utils.data import DataLoader
from src.datasets.audio_dataset import AudioMultiTaskSet
from src.models.multitask import MultiTaskNet

def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

def main():
    cfg={"train_items":"data/lists/deam_train.json",
         "val_items":"data/lists/deam_val.json",
         "lr":1e-3,"bs":2,"epochs":5,"num_genres":1}
    os.makedirs("experiments/logs", exist_ok=True)
    os.makedirs("experiments/checkpoints", exist_ok=True)
    log_csv = "experiments/logs/emotion_curve.csv"
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tr=AudioMultiTaskSet(cfg["train_items"]); va=AudioMultiTaskSet(cfg["val_items"])
    net=MultiTaskNet(num_genres=cfg["num_genres"]).to(dev)
    opt=torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    dl_tr=DataLoader(tr,batch_size=cfg["bs"],shuffle=True)
    dl_va=DataLoader(va,batch_size=cfg["bs"])
    with open(log_csv, "w", newline="") as f:
        w=csv.writer(f); w.writerow(["epoch","rmse_v","rmse_a"])
        for ep in range(cfg["epochs"]):
            net.train()
            for b in dl_tr:
                x=b["spec"].to(dev); e=b["emotion"].to(dev)
                _, le = net(x)
                loss=torch.nn.functional.mse_loss(le,e)
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval(); Ev=[]; Ep=[]
            with torch.no_grad():
                for b in dl_va:
                    x=b["spec"].to(dev); e=b["emotion"].numpy()
                    _, le = net(x); Ev.extend(e); Ep.extend(le.cpu().numpy())
            Ev=np.array(Ev); Ep=np.array(Ep)
            rv=rmse(Ev[:,0],Ep[:,0]); ra=rmse(Ev[:,1],Ep[:,1])
            print(f"[Emotion|Epoch {ep+1}] rmse(V)={rv:.3f} rmse(A)={ra:.3f}")
            w.writerow([ep+1, f"{rv:.6f}", f"{ra:.6f}"]); f.flush()
    torch.save(net.state_dict(), "experiments/checkpoints/emotion_last.pt")

if __name__=="__main__":
    main()
