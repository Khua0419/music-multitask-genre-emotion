# experiments/train_genre_only.py
import os
import csv
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from src.datasets.audio_dataset import AudioMultiTaskSet
from src.models.multitask import MultiTaskNet


def save_confmat_csv(y_true, y_pred, out_csv):
    """Save confusion matrix as CSV (no matplotlib required)."""
    labels = sorted(set(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(int)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + [f"pred_{l}" for l in labels])
        for i, l in enumerate(labels):
            w.writerow([f"true_{l}"] + cm[i].tolist())


def pad_collate(batch):
    """
    Right-pad each spectrogram to the max time length in this batch.
    Expect each item: {"spec": FloatTensor [3, 128, T], "genre": LongTensor()}
    """
    specs = [b["spec"] for b in batch]
    genres = [b["genre"] for b in batch]
    max_T = max(s.shape[-1] for s in specs)

    padded = []
    for s in specs:
        C, F, T = s.shape
        if T == max_T:
            padded.append(s)
        else:
            pad = torch.zeros((C, F, max_T), dtype=s.dtype)
            pad[:, :, :T] = s
            padded.append(pad)

    X = torch.stack(padded, dim=0)                  # [B, 3, 128, max_T]
    G = torch.stack(genres, dim=0).long().view(-1)  # [B]
    return {"spec": X, "genre": G}


def current_lr(optimizer: torch.optim.Optimizer) -> float:
    return float(optimizer.param_groups[0].get("lr", 0.0))


def main():
    cfg = {
    "train_items": "data/lists/gtzan_train.json",
    "val_items":   "data/lists/gtzan_val.json",
    "num_genres":  10,
    "lr":          1e-3,
    "bs":          8,          
    "epochs":      40,        
    "overfit_one_batch_steps": 0,
    "grad_clip":   1.0        
}


    os.makedirs("experiments/logs", exist_ok=True)
    os.makedirs("experiments/checkpoints", exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # ===== Data =====
    tr = AudioMultiTaskSet(cfg["train_items"])
    va = AudioMultiTaskSet(cfg["val_items"])
    print(f"[INFO] len(tr)={len(tr)} len(va)={len(va)}")

    dl_tr = DataLoader(tr, batch_size=cfg["bs"], shuffle=True,  num_workers=0, collate_fn=pad_collate)
    dl_va = DataLoader(va, batch_size=cfg["bs"], shuffle=False, num_workers=0, drop_last=False, collate_fn=pad_collate)

    # ===== Model & Optim =====
    net = MultiTaskNet(num_genres=cfg["num_genres"]).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])

    # LR scheduler (robust)
    sch = None
    sch_is_plateau = False
    try:
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        sch = ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3, threshold=1e-3, cooldown=0)
        sch_is_plateau = True
    except Exception:
        from torch.optim.lr_scheduler import StepLR
        sch = StepLR(opt, step_size=5, gamma=0.5)
        sch_is_plateau = False

    # ===== Overfit-one-batch diagnostics =====
    if cfg["overfit_one_batch_steps"] > 0:
        net.train()
        one = next(iter(dl_tr))
        X = one["spec"].to(dev)
        G = one["genre"].to(dev).long()
        for step in range(1, cfg["overfit_one_batch_steps"] + 1):
            logits, _ = net(X)
            loss = torch.nn.functional.cross_entropy(logits, G)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
            opt.step()
            if step % 10 == 0:
                pred = logits.argmax(1)
                acc = (pred == G).float().mean().item()
                print(f"[OverfitOneBatch] step={step:3d} loss={loss.item():.4f} acc={acc:.3f} lr={current_lr(opt):.2e}")
        torch.save(net.state_dict(), "experiments/checkpoints/genre_overfit_one_batch.pt")
        return

    # ===== Logging =====
    log_csv = "experiments/logs/genre_curve.csv"
    with open(log_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "val_f1", "lr",
                    "gt_set", "pred_set", "pred_hist_0..9", "has_nan"])

        # ===== EPOCH LOOP =====
        for ep in range(1, cfg["epochs"] + 1):
            # ---- Train ----
            net.train()
            tr_loss, tr_correct, tr_count = 0.0, 0, 0
            for b in dl_tr:
                x = b["spec"].to(dev)
                g = b["genre"].to(dev).long()
                logits, _ = net(x)
                loss = torch.nn.functional.cross_entropy(logits, g)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["grad_clip"])
                opt.step()

                tr_loss += float(loss.item())
                tr_correct += (logits.argmax(1) == g).sum().item()
                tr_count += g.numel()

            train_loss = tr_loss / max(len(dl_tr), 1)
            train_acc  = tr_correct / max(tr_count, 1)

            # ---- Validate ----
            net.eval()
            val_loss_sum, val_batches = 0.0, 0
            Gt, Gp = [], []
            has_nan = False
            with torch.no_grad():
                for b in dl_va:
                    x = b["spec"].to(dev)
                    g = b["genre"].to(dev).long()
                    logits, _ = net(x)
                    if torch.isnan(logits).any():
                        has_nan = True
                    vloss = torch.nn.functional.cross_entropy(logits, g)
                    val_loss_sum += float(vloss.item())
                    val_batches += 1

                    pred = logits.argmax(1).cpu().numpy()
                    Gt.extend(g.cpu().numpy().tolist())
                    Gp.extend(pred.tolist())

            val_loss = val_loss_sum / max(val_batches, 1)
            acc = accuracy_score(Gt, Gp)
            f1  = f1_score(Gt, Gp, average="macro", zero_division=0)

            # LR step
            if sch is not None:
                if sch_is_plateau:
                    sch.step(val_loss)
                else:
                    sch.step()

            lr_now = current_lr(opt)

            # Diagnostics: unique sets + histogram of preds
            gt_set = np.unique(np.array(Gt)).tolist()
            pred_set = np.unique(np.array(Gp)).tolist()
            hist = np.bincount(np.array(Gp, dtype=int), minlength=10)[:10].tolist()

            print(
                f"[Genre|Epoch {ep:3d}] "
                f"tr_loss={train_loss:.3f} tr_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.3f} acc={acc:.3f} f1={f1:.3f} lr={lr_now:.2e} | "
                f"gt={gt_set} pred={pred_set} hist={hist} nan={has_nan}"
            )

            w.writerow([
                ep,
                f"{train_loss:.6f}",
                f"{train_acc:.6f}",
                f"{val_loss:.6f}",
                f"{acc:.6f}",
                f"{f1:.6f}",
                f"{lr_now:.6e}",
                str(gt_set),
                str(pred_set),
                str(hist),
                str(has_nan),
            ])
            f.flush()

    # Save final weights & CM
    torch.save(net.state_dict(), "experiments/checkpoints/genre_last.pt")
    save_confmat_csv(Gt, Gp, "experiments/logs/genre_confmat.csv")


if __name__ == "__main__":
    main()
