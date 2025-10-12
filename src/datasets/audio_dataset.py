from pathlib import Path
import json
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from src.features.extract import load_audio, melspec, mfcc, chroma, late_fusion_stack

TARGET_FRAMES = 1300

class AudioMultiTaskSet(Dataset):
    def __init__(self, items_json, sr=22050, cache_feats=True):
        self.items = json.loads(Path(items_json).read_text(encoding="utf-8"))
        self.sr = sr
        self.cache = cache_feats
        self._mem = {}

    def __len__(self): return len(self.items)

    def _extract(self, wav):
        x, sr = load_audio(wav, sr=self.sr)
        feat_np = late_fusion_stack(melspec(x, sr), mfcc(x, sr), chroma(x, sr))  # [C,F,T]
        feat = torch.as_tensor(feat_np, dtype=torch.float32)
        T = feat.shape[-1]
        if T < TARGET_FRAMES:
            feat = F.pad(feat, (0, TARGET_FRAMES - T))
        elif T > TARGET_FRAMES:
            s = (T - TARGET_FRAMES) // 2
            feat = feat[..., s:s+TARGET_FRAMES]
        return feat

    def __getitem__(self, i):
        it = self.items[i]
        k = it["wav"]

        spec = self._mem.get(k) if self.cache else None
        if spec is None:
            spec = self._extract(k)
            if self.cache: self._mem[k] = spec

        # genre: 0..9 for GTZAN; -1 for DEAM
        y = int(it.get("genre", -1))

        # emotion: [val, aro] for DEAM; [nan, nan] placeholder for GTZAN
        emo = it.get("emotion", None)
        if emo is None:
            emo_t = torch.tensor([float("nan"), float("nan")], dtype=torch.float32)
        else:
            if isinstance(emo, (list, tuple)) and len(emo) >= 2:
                emo_t = torch.tensor([float(emo[0]), float(emo[1])], dtype=torch.float32)
            elif isinstance(emo, dict):
                v = float(emo.get("val", emo.get("valence", 0.0)))
                a = float(emo.get("aro", emo.get("arousal", 0.0)))
                emo_t = torch.tensor([v, a], dtype=torch.float32)
            else:
                emo_t = torch.tensor([float("nan"), float("nan")], dtype=torch.float32)

        return {"spec": spec,
                "genre": torch.tensor(y, dtype=torch.long),
                "emotion": emo_t}
