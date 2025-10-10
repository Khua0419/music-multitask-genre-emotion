# scripts/find_good_examples.py
import glob, os, torch, numpy as np
from pathlib import Path
from src.features.extract import load_audio, melspec, mfcc, chroma, late_fusion_stack
from src.models.multitask import MultiTaskNet

ROOT = Path("data/GTZAN_raw")
GENRES = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
CKPT   = r"experiments/checkpoints/genre_last.pt"

def featurize(wav, sr=22050):
    x, _ = load_audio(wav, sr=sr, mono=True)
    mel = melspec(x, sr); mf = mfcc(x, sr); ch = chroma(x, sr)
    spec3 = late_fusion_stack(mel, mf, ch)
    return torch.from_numpy(spec3).unsqueeze(0).float()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = MultiTaskNet(num_genres=len(GENRES)).to(device)
    net.load_state_dict(torch.load(CKPT, map_location=device))
    net.eval()

    hits = []
    with torch.no_grad():
        for gid, gname in enumerate(GENRES):
            files = sorted(glob.glob(str(ROOT / gname / "*.wav")))
            found = None
            for f in files:
                x = featurize(f).to(device)
                logits, _ = net(x)
                pred = int(logits.argmax(1).item())
                if pred == gid:
                    found = f
                    break
            hits.append((gname, found))
            print(f"{gname:>9s} -> {'OK: '+found if found else 'no top1 hit'}")

    print("\n=== Good examples (top1 correct) ===")
    for g, f in hits:
        if f:
            print(f"{g:>9s}: {f}")
    print("\nTip: use one or two of these files in your video demo.")

if __name__ == "__main__":
    main()
