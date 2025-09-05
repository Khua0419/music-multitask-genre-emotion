# Joint Music Emotion & Genre Recognition (Multi-Task + Late Fusion)

## Project Proposal (Feedback One)

### Goal
Build one model that predicts **music genre** (multi-class) and **music emotion** (categorical or arousal–valence) from audio.  
We will compare **single-task vs multi-task** training and evaluate **feature/model late fusion**.

### Why It Matters
Genre and emotion are correlated yet distinct descriptors.  
By training a shared representation with two output heads, we may learn richer music features and improve both tasks.  
This has practical value for music recommendation, emotion-aware applications, and multimodal analysis.

### Datasets
- **GTZAN** (10 genres) for Music Genre Classification (MGC).  
- **DEAM / EmoMusic** for Music Emotion Recognition (MER).  

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

### Deliverables
- Reproducible code on GitHub (with requirements and scripts)  
- A ~10-page project report (following research publication style)  
- A short demo video of the code in action  

---

> **Note**:  
> This repository currently serves as the **Project Proposal submission (Feedback One)**.  
> It contains the project idea, dataset plan, method design, evaluation metrics, and experiment roadmap.  
> Feedback from teaching staff will guide dataset balance, fusion strategy, and metric choices.
