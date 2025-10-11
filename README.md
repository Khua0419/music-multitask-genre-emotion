# Multi-Task Learning with Late-Fusion Features for Music Genre & Emotion

**Repo:** https://github.com/Khua0419/music-multitask-genre-emotion

This stage documents the **GTZAN genre baseline** (real 80/20 split).  
We provide: data prep, training, plots, and a single-file inference demo.

---

## Environment

```bash
conda create -n mtl-audio python=3.10 -y
conda activate mtl-audio
pip install -r requirements.txt
data/GTZAN_raw/<genre>/<file>.wav

