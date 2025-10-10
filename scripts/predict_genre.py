# scripts/predict_genre.py
import sys, torch, numpy as np
from src.features.extract import load_audio, melspec, mfcc, chroma, late_fusion_stack
from src.models.multitask import MultiTaskNet

GENRES = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
CKPT   = r"experiments/checkpoints/genre_last.pt"  # 方案A：直接用last

def featurize(wav, sr=22050):
    x, _ = load_audio(wav, sr=sr, mono=True)
    mel = melspec(x, sr); mf = mfcc(x, sr); ch = chroma(x, sr)
    spec3 = late_fusion_stack(mel, mf, ch)          # [3,F,T]
    return torch.from_numpy(spec3).unsqueeze(0).float()  # [1,3,F,T]

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m scripts.predict_genre <wav_path> [topk]")
        return
    wav  = sys.argv[1]
    topk = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = MultiTaskNet(num_genres=len(GENRES)).to(device)
    net.load_state_dict(torch.load(CKPT, map_location=device))
    net.eval()

    x = featurize(wav).to(device)
    with torch.no_grad():
        logits, _ = net(x)
        prob = torch.softmax(logits, dim=1).cpu().numpy()[0]
    idx = np.argsort(prob)[::-1][:topk]

    print(f"\nTop-{topk} for {wav}:")
    for i in idx:
        print(f"  {GENRES[i]:>9s}  p={prob[i]:.3f}")

if __name__ == "__main__":
    main()
