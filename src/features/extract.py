import numpy as np, librosa
def load_audio(path, sr=22050, mono=True):
    x, fs = librosa.load(path, sr=sr, mono=mono)
    x = librosa.util.normalize(x); return x, fs
def melspec(x, sr, n_mels=128, hop=512, n_fft=1024):
    S = librosa.feature.melspectrogram(y=x, sr=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
    return librosa.power_to_db(S, ref=np.max).astype(np.float32)
def mfcc(x, sr, n_mfcc=20, hop=512, n_fft=1024):
    return librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc, hop_length=hop, n_fft=n_fft).astype(np.float32)
def chroma(x, sr, hop=512, n_fft=1024):
    return librosa.feature.chroma_stft(y=x, sr=sr, hop_length=hop, n_fft=n_fft).astype(np.float32)
def late_fusion_stack(mel, mf, ch):
    F=max(mel.shape[0], mf.shape[0], ch.shape[0]); T=min(mel.shape[1], mf.shape[1], ch.shape[1])
    def pad(A):
        if A.shape[0]==F: return A[:, :T]
        P=np.zeros((F,T),dtype=A.dtype); P[:A.shape[0],:T]=A[:,:T]; return P
    return np.stack([pad(mel), pad(mf), pad(ch)], axis=0)
