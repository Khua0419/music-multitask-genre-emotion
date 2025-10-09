import torch, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.datasets.audio_dataset import AudioMultiTaskSet
from src.models.multitask import MultiTaskNet

def main():
    cfg={"train_items":"data/lists/gtzan_train.json",
         "val_items":"data/lists/gtzan_val.json",
         "num_genres":10, "lr":1e-3, "bs":2, "epochs":3}
    dev="cuda" if torch.cuda.is_available() else "cpu"
    tr=AudioMultiTaskSet(cfg["train_items"]); va=AudioMultiTaskSet(cfg["val_items"])
    net=MultiTaskNet(num_genres=cfg["num_genres"]).to(dev)
    opt=torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    dl_tr=DataLoader(tr,batch_size=cfg["bs"],shuffle=True)
    dl_va=DataLoader(va,batch_size=cfg["bs"])
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
        acc=accuracy_score(Gt,Gp); f1=f1_score(Gt,Gp,average="macro", zero_division=0)
        print(f"[Genre|Epoch {ep+1}] acc={acc:.3f} f1={f1:.3f}")

if __name__=="__main__":
    main()
