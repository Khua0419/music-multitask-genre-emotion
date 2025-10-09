from pathlib import Path
import json, torch
from torch.utils.data import Dataset
from src.features.extract import load_audio, melspec, mfcc, chroma, late_fusion_stack

class AudioMultiTaskSet(Dataset):
    def __init__(self, items_json, sr=22050, cache_feats=True):
        self.items = json.loads(Path(items_json).read_text(encoding="utf-8"))
        self.sr = sr; self.cache = cache_feats; self._mem = {}
    def __len__(self): return len(self.items)
    def _extract(self, wav):
        x, sr = load_audio(wav, sr=self.sr)
        return late_fusion_stack(melspec(x,sr), mfcc(x,sr), chroma(x,sr))
    def __getitem__(self, i):
        it = self.items[i]; k = it["wav"]
        spec3 = self._mem.get(k) if self.cache else None
        if spec3 is None:
            spec3 = self._extract(k)
            if self.cache: self._mem[k]=spec3
        return {"spec": torch.tensor(spec3, dtype=torch.float32),
                "genre": torch.tensor(it["genre"], dtype=torch.long),
                "emotion": torch.tensor(it["emotion"], dtype=torch.float32)}
