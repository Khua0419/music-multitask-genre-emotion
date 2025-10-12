import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from src.features.extract import load_audio, melspec, mfcc, chroma, late_fusion_stack
from src.models.multitask import MultiTaskNet
from src.datasets.audio_dataset import TARGET_FRAMES  # keep same as training

LABELS_10 = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

def find_audio(p: Path):
    """Return a list of audio files for a given file or directory."""
    if p.is_file():
        return [p]
    exts = {".wav",".mp3",".flac",".ogg",".m4a"}
    return sorted([x for x in p.rglob("*") if x.suffix.lower() in exts])

def extract_feat(wav_path: str, sr: int = 22050) -> torch.Tensor:
    """
    Load audio and compute late-fusion features [C, F, T], then
    pad/center-crop along time to TARGET_FRAMES (same as training).
    """
    x, sr = load_audio(wav_path, sr=sr)
    feat_np = late_fusion_stack(melspec(x, sr), mfcc(x, sr), chroma(x, sr))  # [C, F, T]
    feat = torch.tensor(feat_np, dtype=torch.float32)
    T = feat.shape[-1]
    if T < TARGET_FRAMES:
        feat = F.pad(feat, (0, TARGET_FRAMES - T))  # right-pad on time axis
    elif T > TARGET_FRAMES:
        s = (T - TARGET_FRAMES) // 2                 # center-crop
        feat = feat[..., s:s+TARGET_FRAMES]
    return feat

def extract_feat_crops(wav_path: str, sr: int = 22050, crops: int = 3):
    """
    Make N time-crops of length TARGET_FRAMES from the feature [C,F,T].
    If audio is short or crops<=1, return a single padded/center-cropped window.
    """
    x, sr = load_audio(wav_path, sr=sr)
    feat_np = late_fusion_stack(melspec(x, sr), mfcc(x, sr), chroma(x, sr))  # [C, F, T]
    feat = torch.tensor(feat_np, dtype=torch.float32)
    T = feat.shape[-1]

    if T <= TARGET_FRAMES or crops <= 1:
        if T < TARGET_FRAMES:
            feat = F.pad(feat, (0, TARGET_FRAMES - T))
        elif T > TARGET_FRAMES:
            s = (T - TARGET_FRAMES) // 2
            feat = feat[..., s:s + TARGET_FRAMES]
        return [feat]

    starts = np.linspace(0, T - TARGET_FRAMES, num=crops).astype(int).tolist()
    return [feat[..., s:s + TARGET_FRAMES] for s in starts]

def main():
    ap = argparse.ArgumentParser("Predict genre + [valence, arousal]")
    ap.add_argument("--ckpt", required=True, help="Path to mtl_best_*.pt or mtl_last.pt")
    ap.add_argument("--wav",  required=True, help="Audio file or directory")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--tta",  type=int, default=1, help="number of time-crops to average; 1 disables TTA")
    args = ap.parse_args()

    # Load checkpoint and config
    ckpt = torch.load(args.ckpt, map_location="cpu")
    cfg = ckpt.get("cfg", {})
    num_genres = int(cfg.get("num_genres", 10))
    labels = LABELS_10 if num_genres == 10 else [f"class_{i}" for i in range(num_genres)]

    # Build model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = MultiTaskNet(num_genres=num_genres).to(device)
    net.load_state_dict(ckpt["model"], strict=True)
    net.eval()

    # Collect audio files
    files = find_audio(Path(args.wav))
    if not files:
        print("No audio found. Check --wav path.")
        return

    # Inference (with optional TTA)
    for p in files:
        if args.tta <= 1:
            with torch.no_grad():
                spec = extract_feat(str(p)).unsqueeze(0).to(device)  # [1, C, F, T]
                logits, emo = net(spec)                              # [1, G], [1, 2]
                probs_avg = F.softmax(logits, dim=1).cpu().numpy()[0]
                emo_avg = emo.cpu().numpy()[0]
        else:
            crops = extract_feat_crops(str(p), crops=args.tta)       # list of [C, F, T]
            batch = torch.stack(crops, 0).to(device)                 # [N, C, F, T]
            with torch.no_grad():
                logits, emo = net(batch)                             # [N, G], [N, 2]
                probs = F.softmax(logits, dim=1).cpu().numpy()       # [N, G]
                probs_avg = probs.mean(axis=0)                       # [G]
                emo_avg = emo.cpu().numpy().mean(axis=0)             # [2]

        # Top-k on averaged probs
        topk = min(args.topk, num_genres)
        idx = np.argsort(-probs_avg)[:topk]
        va = emo_avg.tolist()

        # Print results
        print(f"\nFile: {p}")
        print("  Top-{} genre:".format(topk))
        for i in idx:
            name = labels[i] if i < len(labels) else f"class_{i}"
            print(f"    - {name:>10s}: {probs_avg[i]:.3f}")
        print(f"  Emotion: valence={va[0]:.3f}, arousal={va[1]:.3f}")

if __name__ == "__main__":
    main()
