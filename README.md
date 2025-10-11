# 🎵 Multi-Task Learning
### Late-Fusion Features for Music Genre & Emotion

**Repo:** [https://github.com/Khua0419/music-multitask-genre-emotion](https://github.com/Khua0419/music-multitask-genre-emotion)

This project explores multi-task learning for music understanding, combining genre classification and emotion regression within a single deep-learning framework.
Instead of training two independent models, we adopt a shared CNN encoder that jointly learns timbral and rhythmic representations from audio spectrograms, followed by task-specific heads for genre (categorical prediction) and emotion (continuous valence–arousal estimation).
By training on both GTZAN (genre) and DEAM (emotion) datasets, the model leverages complementary cues—genre structure often correlates with emotional tone—to improve overall generalization.
The repository includes:
- Individual baselines for GTZAN and DEAM
- A unified multi-task network with late-fusion features
- Scripts for preprocessing, training, evaluation, and visualization

This work serves as a foundation for further research on cross-domain music representation learning and affective computing.

---

## 🧩 Environment

```bash
conda create -n mtl-audio python=3.10 -y
conda activate mtl-audio
pip install -r requirements.txt
```
Data and checkpoints are git-ignored (data/**, experiments/checkpoints/**).

## 🧠 Genre-Only Baseline (GTZAN)

### 1. Overview
This section presents the genre classification baseline on the GTZAN dataset.
The dataset consists of 10 balanced genres (100 tracks per genre), each with 30-second 22.05 kHz mono `.wav` files.
We start from a lightweight CNN baseline and evaluate classification accuracy and F1-score on validation data.

### 2. Data Layout
```bash
data/GTZAN_raw/
  ├── blues/
  ├── classical/
  ├── country/
  ├── disco/
  ├── hiphop/
  ├── jazz/
  ├── metal/
  ├── pop/
  ├── reggae/
  └── rock/
```
Each folder contains 100 audio clips per genre (GTZAN standard structure).

### 3. Preprocessing
Step 1 — Generate stratified 80/20 train/val splits
```bash
python scripts/make_full_splits_balanced.py
```
Creates:
```bash
data/lists/gtzan_train.json
data/lists/gtzan_val.json
```
Step 2 — Extract Mel-spectrograms (optional if already cached)
```bash
python -m scripts.extract_mels_gtzan
```
- 22.05 kHz mono audio
- 128 Mel bins, `n_fft = 1024`, `hop = 512`
- Normalized to `[0, 1]` and stored as `.npy`

### 4. Model Architecture
- CNN Baseline
  - 3 convolutional blocks with batch normalization and ReLU
  - Global average pooling + fully connected classifier (10 genres)
  - Trained using cross-entropy loss, metrics: accuracy and F1-score
  - Lightweight and fast, designed as a baseline for multitask extension
 
### 5. Training & Evaluation
Training command
```bash
python -m experiments.train_genre_only
```
Key outputs (after training):
```bash
experiments/logs/genre_curve.csv
experiments/logs/genre_curve.png
experiments/logs/genre_confmat.csv
experiments/logs/genre_confmat.png
experiments/checkpoints/genre_last.pt
```
### 6. Validation Plots
#### Acc/F1 Learning Curve (validation)
![Genre curve](experiments/logs/genre_curve.png)

#### Confusion Matrix (validation)
![Genre confmat](experiments/logs/genre_confmat.png)

### 7. Inference
Single-file prediction (Top-k)
```bash
python -m scripts.predict_genre "data/GTZAN_raw/jazz/jazz.00000.wav" 5
```
Sample output
```bash
Top-5 for data/GTZAN_raw/jazz/jazz.00000.wav:
classical  p=0.835
jazz       p=0.069
country    p=0.050
blues      p=0.030
reggae     p=0.008
```

### 8. Summary
- The GTZAN CNN baseline achieves stable convergence across 10 genres with balanced training.
- Validation accuracy and F1-score curves confirm model stability after ~40 epochs.
- Confusion matrix highlights overlap between musically similar genres (e.g., jazz–blues, pop–disco).
- This baseline provides the Genre branch for later multitask training with emotion regression (DEAM).

---

## 🎧 Emotion-Only Baseline (DEAM)

### 1. Overview
This section describes our **emotion regression baseline** on the [DEAM dataset](https://cvml.unige.ch/databases/DEAM/), predicting continuous **Valence** and **Arousal** values in the range `[1, 9]`.  
We first built a light CNN model as the baseline, and then introduced a stronger **CRNN (CNN + BiGRU)** architecture to capture temporal emotion dynamics.

### 2. Data Layout
```bash
data/DEAM/
  audio/             # Original DEAM audio files (mp3/wav)
  annotations/       # Official annotation CSVs
  mels/              # Auto-generated Mel-spectrograms (.npy)
data/lists/
  deam_train.json    # Song-level training split (80%)
  deam_val.json      # Song-level validation split (20%)
  deam_train_mel.json / deam_val_mel.json

```
### 3. Preprocessing
Step 1 — Generate splits
```bash
python -m scripts.make_deam_splits
```
Step 2 — Extract Mel-spectrograms
```bash
python -m scripts.extract_mels_deam
```
Step 2 — Extract Mel-spectrograms
```bash
python -m scripts.extract_mels_deam
```
- 22.05 kHz mono audio  
- 128 mel bins, `n_fft = 1024`, `hop = 512`  
- Normalized to `[0, 1]` and saved as `.npy`

### 4. Model Architectures
- **CNN Baseline**  
  A compact 3-layer CNN followed by global average pooling and a 2-unit regression head.  
  Time information is mostly flattened — simple and efficient but limited in modeling temporal emotion flow.

- **CRNN (CNN + BiGRU)**  
  To capture temporal dependencies, the CRNN uses:  
  - 3 convolutional blocks to extract short-time features  
  - Mean-frequency pooling → transforms `[B, C, F′, T′]` into `[B, T′, C]`  
  - A bidirectional GRU (`hidden = 128`, `layers = 1`) to learn emotional evolution  
  - Temporal mean pooling + linear layer → output `[Valence, Arousal]`

### 5. Training & Evaluation

**Training command**
```bash
python -m experiments.train_deam_crnn
```
| Model           | Val MSE ↓ | Val Pearson ↑ |
|:----------------|----------:|--------------:|
| CNN Baseline    | ≈ 0.91    | 0.63          |
| **CRNN (BiGRU)**| **≈ 0.62**| **0.78**      |

**Learning curve**

<p align="center">
  <img src="experiments/logs/deam_crnn_curve.png" width="70%">
</p>

### 6. Inference
Single-file prediction
```bash
python -m scripts.predict_emotion "data/DEAM/mels/1000.npy"
```
Output example:
```bash
Emotion (Valence, Arousal): 5.46, 5.50
```
Sliding-window averaging (more stable)
```bash
python -m scripts.predict_emotion_window "data/DEAM/mels/1000.npy"
```

### 7. Summary
- The CRNN significantly improves temporal emotion modeling, boosting Pearson r from 0.63 → 0.78.
- The architecture remains lightweight and fully compatible with the genre-emotion multitask pipeline.
- Future work will extend this module into a shared-encoder multitask framework, enabling simultaneous learning of genre + emotion representations.
