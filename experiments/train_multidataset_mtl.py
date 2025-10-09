import csv, os, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.datasets.audio_dataset import AudioMultiTaskSet
from src.models.multitask import MultiTaskNet

def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

def main():
    cfg={"gt_train":"data/lists/gtzan_train.json","gt_val":"data/lists/gtzan_val.json",
         "de_train":"data/lists/deam_train.json","de_val":"data/lists/deam_val.json",
         "num_genres":10,"lr":1e-3,"bs":2,"epochs":5,"lam_g":1.0,"lam_e":1.0}
    os.makedirs("experiments/logs", exist_ok=True)
    os.makedirs("experiments/checkpoints", exist_ok=True)
    log_csv = "experiments/logs/mtl_curve.csv"
    dev="cuda" if torch.cuda.is_available() else "cpu"
    Gtr=AudioMultiTaskSet(cfg["gt_train"]); Gva=AudioMultiTaskSet(cfg["gt_val"])
    Dtr=AudioMultiTaskSet(cfg["de_train"]); Dva=AudioMultiTaskSet(cfg["de_val"])
    net=MultiTaskNet(num_genres=cfg["num_genres"]).to(dev)
    opt=torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    dl_g=DataLoader(Gtr,batch_size=cfg["bs"],shuffle=True)
    dl_d=DataLoader(Dtr,batch_size=cfg["bs"],shuffle=True)
    with open(log_csv, "w", newline="") as f:
        w=csv.writer(f); w.writerow(["epoch","acc","f1","rmse_v","rmse_a"])
        for ep in range(cfg["epochs"]):
            net.train()
            it_g, it_d = iter(dl_g), iter(dl_d)
            steps=max(len(dl_g),len(dl_d))
            for _ in range(steps):
                try: bg=next(it_g)
                except StopIteration: it_g=iter(dl_g); bg=next(it_g)
                try: bd=next(it_d)
                except StopIteration: it_d=iter(dl_d); bd=next(it_d)
                xg=bg["spec"].to(dev); gg=bg["genre"].to(dev)
                lg,_=net(xg); loss_g=torch.nn.functional.cross_entropy(lg,gg)
                xd=bd["spec"].to(dev); ee=bd["emotion"].to(dev)
                _,le=net(xd); loss_e=torch.nn.functional.mse_loss(le,ee)
                loss = cfg["lam_g"]*loss_g + cfg["lam_e"]*loss_e
                opt.zero_grad(); loss.backward(); opt.step()
            # val
            net.eval(); Gt=[]; Gp=[]; Ev=[]; Ep=[]
            with torch.no_grad():
                for b in DataLoader(Gva,batch_size=cfg["bs"]):
                    x=b["spec"].to(dev); g=b["genre"].numpy()
                    lg,_=net(x); Gt.extend(g); Gp.extend(lg.argmax(1).cpu().numpy())
                for b in DataLoader(Dva,batch_size=cfg["bs"]):
                    x=b["spec"].to(dev); e=b["emotion"].numpy()
                    _,le=net(x); Ev.extend(e); Ep.extend(le.cpu().numpy())
            acc=accuracy_score(Gt,Gp); f1=f1_score(Gt,Gp,average="macro",zero_division=0)
            Ev=np.array(Ev); Ep=np.array(Ep)
            rv=float(np.sqrt(np.mean((Ev[:,0]-Ep[:,0])**2)))
            ra=float(np.sqrt(np.mean((Ev[:,1]-Ep[:,1])**2)))
            print(f"[MTL|Epoch {ep+1}] acc={acc:.3f} f1={f1:.3f} rmse(V)={rv:.3f} rmse(A)={ra:.3f}")
            w.writerow([ep+1, f"{acc:.6f}", f"{f1:.6f}", f"{rv:.6f}", f"{ra:.6f}"]); f.flush()
    torch.save(net.state_dict(), "experiments/checkpoints/mtl_last.pt")

if __name__=="__main__":
    main()