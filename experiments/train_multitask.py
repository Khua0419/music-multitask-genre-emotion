import json, argparse, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.datasets.audio_dataset import AudioMultiTaskSet
from src.models.multitask import MultiTaskNet
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))

def main(cfg_path):
    cfg = json.loads(open(cfg_path,"r").read())
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = AudioMultiTaskSet(cfg["train_items"]); va = AudioMultiTaskSet(cfg["val_items"])
    net = MultiTaskNet(num_genres=cfg["num_genres"]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.get("lr",1e-3))
    dl_tr = DataLoader(tr, batch_size=cfg.get("bs",2), shuffle=True)
    dl_va = DataLoader(va, batch_size=cfg.get("bs",2))
    for ep in range(cfg.get("epochs",2)):
        net.train()
        for b in dl_tr:
            x=b["spec"].to(dev); g=b["genre"].to(dev); e=b["emotion"].to(dev)
            lg, le = net(x)
            loss = torch.nn.functional.cross_entropy(lg,g) + torch.nn.functional.mse_loss(le,e)
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval(); Gt=[]; Gp=[]; Ev=[]; Ep=[]
        with torch.no_grad():
            for b in dl_va:
                x=b["spec"].to(dev); g=b["genre"].numpy(); e=b["emotion"].numpy()
                lg, le = net(x)
                Gt.extend(g); Gp.extend(lg.argmax(1).cpu().numpy())
                Ev.extend(e);  Ep.extend(le.cpu().numpy())
        acc=accuracy_score(Gt,Gp); f1=f1_score(Gt,Gp,average="macro")
        Ev=np.array(Ev); Ep=np.array(Ep)
        print(f"[Epoch {ep+1}] acc={acc:.3f} f1={f1:.3f} rmse(V)={rmse(Ev[:,0],Ep[:,0]):.3f} rmse(A)={rmse(Ev[:,1],Ep[:,1]):.3f}")

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--cfg", required=True)
    main(ap.parse_args().cfg)
