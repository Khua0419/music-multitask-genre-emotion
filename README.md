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
```
## Data (GTZAN)

Place files like:
```bash
data/GTZAN_raw/<genre>/<file>.wav
```
Create stratified 80/20 train/val lists:
```bash
python scripts/make_full_splits_balanced.py
# creates:
#   data/lists/gtzan_train.json
#   data/lists/gtzan_val.json
```
Train (Genre baseline)
```bash
python -m experiments.train_genre_only
```
The script writes logs to:

experiments/logs/genre_curve.csv (per-epoch val metrics)

experiments/logs/genre_confmat.csv (confusion matrix of the best epoch)

optional checkpoint(s) in experiments/checkpoints/ (git-ignored)
