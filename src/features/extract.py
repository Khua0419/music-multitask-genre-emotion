# src/features/extract.py
from __future__ import annotations

import numpy as np
import librosa
import soundfile as sf

__all__ = [
    "load_audio",
    "melspec",
    "mfcc",
    "chroma",
    "late_fusion_stack",
]


def load_audio(path: str, sr: int = 22050, mono: bool = True):
    """
    Try soundfile first (fast & precise). If it fails (non-standard WAV, etc.),
    fall back to librosa.load (uses audioread/ffmpeg).
    """
    try:
        # 1) primary: soundfile (keeps original sample rate)
        x, sr0 = sf.read(path, dtype="float32", always_2d=False)
        if mono and np.ndim(x) > 1:
            x = x.mean(axis=1)
        if sr0 != sr:
            x = librosa.resample(x, orig_sr=sr0, target_sr=sr)
        x = librosa.util.normalize(x)
        return x, sr
    except Exception:
        # 2) fallback: librosa (+audioread/ffmpeg)
        x, fs = librosa.load(path, sr=sr, mono=mono)
        x = librosa.util.normalize(x)
        return x, fs


def melspec(x: np.ndarray, sr: int, n_mels: int = 128, hop: int = 512, n_fft: int = 1024) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)


def mfcc(x: np.ndarray, sr: int, n_mfcc: int = 20, hop: int = 512, n_fft: int = 1024) -> np.ndarray:
    M = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc, hop_length=hop, n_fft=n_fft)
    return M.astype(np.float32)


def chroma(x: np.ndarray, sr: int, hop: int = 512, n_fft: int = 1024) -> np.ndarray:
    C = librosa.feature.chroma_stft(y=x, sr=sr, hop_length=hop, n_fft=n_fft)
    return C.astype(np.float32)


def late_fusion_stack(mel: np.ndarray, mf: np.ndarray, ch: np.ndarray) -> np.ndarray:
    """
    Stack [mel, mfcc, chroma] → [3, F, T]
    Pad/truncate along F to the max F, and along T to the min T.
    """
    F = max(mel.shape[0], mf.shape[0], ch.shape[0])
    T = min(mel.shape[1], mf.shape[1], ch.shape[1])

    def pad(A: np.ndarray) -> np.ndarray:
        if A.shape[0] == F:
            return A[:, :T]
        P = np.zeros((F, T), dtype=A.dtype)
        P[: A.shape[0], :T] = A[:, :T]
        return P

    return np.stack([pad(mel), pad(mf), pad(ch)], axis=0)
