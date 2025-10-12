import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

from src.datasets.audio_dataset import AudioMultiTaskSet
from src.models.multitask import MultiTaskNet


def rmse(a, b) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def main(cfg_path: str):
    # -------------------- config & seeds --------------------
    cfg = json.loads(open(cfg_path, "r", encoding="utf-8").read())
    np.random.seed(42)
    torch.manual_seed(42)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = int(cfg.get("bs", 2))
    epochs = int(cfg.get("epochs", 2))
    lr = float(cfg.get("lr", 1e-3))
    lam_genre = float(cfg.get("lam_genre", 1.5)); lam_emo = float(cfg.get("lam_emo", 1.0))  # CE weight / MSE weight

    # optional overrideable paths
    logs_dir = cfg.get("logs_dir", "experiments/logs")
    ckpt_dir = cfg.get("ckpt_dir", "experiments/checkpoints")
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, "mtl_curve.csv")
    if not os.path.exists(log_path):
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("epoch,lr,acc,f1,rmse_v,rmse_a,score\n")

    # -------------------- data --------------------
    # num_workers=0 is safest on Windows
    tr = AudioMultiTaskSet(cfg["train_items"])
    va = AudioMultiTaskSet(cfg["val_items"])
    dl_tr = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=0)
    dl_va = DataLoader(va, batch_size=bs, shuffle=False, num_workers=0)

    # -------------------- model & optim --------------------
    net = MultiTaskNet(num_genres=cfg["num_genres"]).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # LR scheduler (older PyTorch has no `verbose` argument)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=3
    )
    best_score = float("inf")

    # -------------------- training loop --------------------
    for ep in range(epochs):
        net.train()
        for b in dl_tr:
            x = b["spec"].to(device)
            g = b["genre"].to(device)      # -1 means "no genre label" (DEAM items)
            e = b["emotion"].to(device)    # [nan, nan] means "no emotion label" (GTZAN items)

            lg, le = net(x)                # lg: (B, num_classes), le: (B, 2)

            # Masked multitask loss
            mask_cls = (g != -1)
            mask_reg = torch.isfinite(e).all(dim=1)

            ce  = F.cross_entropy(lg[mask_cls], g[mask_cls]) if mask_cls.any() else (lg.sum() * 0.0)
            mse = F.mse_loss(le[mask_reg], e[mask_reg])       if mask_reg.any() else (le.sum() * 0.0)
            loss = lam_genre * ce + lam_emo * mse

            opt.zero_grad()
            loss.backward()
            opt.step()

        # -------------------- validation --------------------
        net.eval()
        Gt, Gp, Ev, Ep = [], [], [], []
        with torch.no_grad():
            for b in dl_va:
                x = b["spec"].to(device)
                g = b["genre"].cpu().numpy()
                e = b["emotion"].cpu().numpy()

                lg, le = net(x)
                Gt.extend(g)
                Gp.extend(lg.argmax(1).cpu().numpy())
                Ev.extend(e)
                Ep.extend(le.cpu().numpy())

        # classification metrics on samples that actually have labels
        Gt = np.asarray(Gt)
        Gp = np.asarray(Gp)
        mask_cls_val = (Gt != -1)
        acc = accuracy_score(Gt[mask_cls_val], Gp[mask_cls_val]) if mask_cls_val.any() else 0.0
        f1  = f1_score(Gt[mask_cls_val], Gp[mask_cls_val], average="macro", zero_division=0) if mask_cls_val.any() else 0.0

        # regression metrics on samples that actually have VA labels
        Ev = np.asarray(Ev, dtype=float)
        Ep = np.asarray(Ep, dtype=float)
        mask_reg_val = np.isfinite(Ev).all(axis=1)
        rmse_v = rmse(Ev[mask_reg_val, 0], Ep[mask_reg_val, 0]) if mask_reg_val.any() else float("nan")
        rmse_a = rmse(Ev[mask_reg_val, 1], Ep[mask_reg_val, 1]) if mask_reg_val.any() else float("nan")

        # combined validation score (lower is better)
        avg_rmse = float(np.nanmean([rmse_v, rmse_a])) if not (np.isnan(rmse_v) and np.isnan(rmse_a)) else 0.0
        score = (1.0 - f1) + avg_rmse

        cur_lr = opt.param_groups[0]["lr"]
        print(f"[Epoch {ep+1}] lr={cur_lr:.6f} acc={acc:.3f} f1={f1:.3f} rmse(V)={rmse_v:.3f} rmse(A)={rmse_a:.3f}")

        # logging
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{ep+1},{cur_lr:.8f},{acc:.4f},{f1:.4f},{rmse_v:.6f},{rmse_a:.6f},{score:.6f}\n")

        # checkpoints: always save last; keep best by score
        torch.save({"model": net.state_dict(), "cfg": cfg, "epoch": ep+1},
                   os.path.join(ckpt_dir, "mtl_last.pt"))
        if score < best_score:
            best_score = score
            torch.save(
                {
                    "model": net.state_dict(),
                    "cfg": cfg,
                    "epoch": ep+1,
                    "val": {"acc": acc, "f1": f1, "rmse_v": rmse_v, "rmse_a": rmse_a},
                },
                os.path.join(ckpt_dir, f"mtl_best_ep{ep+1}.pt"),
            )

        # step LR scheduler with the validation score
        sched.step(score)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    main(ap.parse_args().cfg)
