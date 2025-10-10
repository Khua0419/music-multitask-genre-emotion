# scripts/eval_genre_checkpoint.py
import torch, numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
from src.datasets.audio_dataset import AudioMultiTaskSet
from src.models.multitask import MultiTaskNet
import torch.nn.functional as F

VAL = r"data/lists/gtzan_val.json"
CKPT = r"experiments/checkpoints/genre_last.pt"
BS = 8

def pad_collate(batch):
    specs = [b["spec"] for b in batch]
    labels= [b["genre"] for b in batch]
    T = max(s.shape[-1] for s in specs)
    out = [F.pad(s, (0, T - s.shape[-1])) if s.shape[-1] < T else s[..., :T] for s in specs]
    X = torch.stack(out, 0)
    y = torch.stack(labels, 0).long().view(-1)
    return {"spec": X, "genre": y}

def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    va = AudioMultiTaskSet(VAL)
    dl = DataLoader(va, batch_size=BS, shuffle=False, num_workers=0, collate_fn=pad_collate)
    net = MultiTaskNet(num_genres=10).to(dev)
    net.load_state_dict(torch.load(CKPT, map_location=dev))
    net.eval()
    Gt, Gp = [], []
    with torch.no_grad():
        for b in dl:
            x = b["spec"].to(dev)
            y = b["genre"].cpu().numpy()
            lg, _ = net(x)
            p = lg.argmax(1).cpu().numpy()
            Gt.extend(y); Gp.extend(p)
    print(f"VAL | Acc={accuracy_score(Gt,Gp):.3f}  F1={f1_score(Gt,Gp,average='macro'):.3f}  (N={len(Gt)})")

if __name__ == "__main__":
    main()
