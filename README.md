# Joint Music Emotion & Genre Recognition (Multi-Task + Late Fusion)

## Project Proposal (Feedback One)

### Goal
Build one model that predicts **music genre** (multi-class) and **music emotion** (categorical or arousal–valence) from audio.  
We will compare **single-task vs multi-task** training and evaluate **feature/model late fusion**.
**Research Question**: Can multi-task learning with late fusion improve both genre classification and emotion recognition compared to single-task baselines?

### Why It Matters
Genre and emotion are correlated yet distinct descriptors.  
By training a shared representation with two output heads, we may learn richer music features and improve both tasks.  
This has practical value for music recommendation, emotion-aware applications, and multimodal analysis.

### Datasets
- **GTZAN** (10 genres) for Music Genre Classification (MGC).  
- **DEAM / EmoMusic** for Music Emotion Recognition (MER).
- **GTZAN**: Contains 1,000 audio tracks (~30s each), evenly distributed across 10 genres (100 tracks per genre).  
**DEAM / EmoMusic**: Contains ~1,500 songs with continuous or categorical emotion annotations (arousal–valence or discrete classes).  

These datasets are widely used in Music Information Retrieval (MIR) research and provide a standard benchmark for reproducibility.

Clips will be segmented into 10–30 seconds, and standard audio augmentations (gain, time-shift, SpecAugment, noise mixing) will be applied.

### Method
- **Features**: 128-Mel spectrogram (+ optional MFCC / Chroma)  
- **Backbone**: CRNN (Conv2D → BiLSTM) or Transformer Encoder  
- **Outputs**: 
  - Head 1 → Genre classification (softmax)  
  - Head 2 → Emotion classification (softmax) or AV regression (MSE loss)  
- **Loss**: `L_total = L_genre + λ · L_emotion` (λ is tunable)  
- **Fusion**: Late fusion across feature branches (Mel, MFCC, Chroma).  

### Planned Experiments
1. Single-task vs Multi-task training  
2. Mel vs Mel+MFCC/Chroma (late fusion)  
3. BiLSTM vs Transformer backbone  
4. Transfer: pretrain on Genre → finetune Emotion  

### Evaluation Metrics
- **Genre**: Accuracy, Macro-F1, Confusion Matrix  
- **Emotion (categorical)**: Macro-F1, ROC-AUC  
- **Emotion (AV regression)**: RMSE, CCC
- - **Genre**: Accuracy, Macro-F1, Confusion Matrix  
  *Rationale*: Accuracy measures overall performance, while Macro-F1 addresses potential class imbalance across genres.
- **Emotion (categorical)**: Macro-F1, ROC-AUC  
  *Rationale*: F1 balances precision and recall, and ROC-AUC evaluates classifier robustness across thresholds.
- **Emotion (AV regression)**: RMSE, CCC  
  *Rationale*: RMSE reflects prediction error magnitude, while Concordance Correlation Coefficient (CCC) captures both correlation and scale agreement with ground truth.


### Deliverables
- Reproducible code on GitHub (with requirements and scripts)  
- A ~10-page project report (following research publication style)  
- A short demo video of the code in action  

---

> **Note**:  
> This repository currently serves as the **Project Proposal submission (Feedback One)**.  
> It contains the project idea, dataset plan, method design, evaluation metrics, and experiment roadmap.  
> Feedback from teaching staff will guide dataset balance, fusion strategy, and metric choices.
