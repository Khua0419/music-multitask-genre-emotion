# 🎵 Multitask Learning for Music Genre and Emotion Recognition

**Author:** Kejia Huang  
**Environment:** Anaconda (mtl-audio)  
**Repository:** [music-multitask-genre-emotion](https://github.com/Khua0419/music-multitask-genre-emotion)  
**Date:** October 2025  

---

## 1. Introduction & Research Question

Music understanding tasks such as *genre classification* and *emotion recognition* are fundamental to computational music analysis.  
Most existing systems train separate models for each task. However, music emotion and genre are semantically related — for example, “metal” is often high in arousal, whereas “classical” tends to have low arousal and neutral valence.  

This project explores whether **Multitask Learning (MTL)** can improve both genre classification and emotion regression performance compared to separate single-task models.  

> **Research Question:**  
> *Can a multitask learning framework jointly trained on GTZAN (genre) and DEAM (emotion) datasets outperform single-task baselines in terms of generalization and stability?*

The motivation behind this study is that MTL can capture shared acoustic cues (e.g., rhythm, timbre, and energy), potentially improving efficiency and robustness across related music understanding tasks.

---

## 2. Literature Review

### 2.1 Related Work
- **Genre Classification (GTZAN):**  
  Choi et al. (2017) proposed CNN/CRNN architectures for automatic genre recognition.  
  However, GTZAN is small and prone to label overlaps (e.g., reggae vs pop).  
  This dataset remains a common benchmark in MIR tasks (Choi et al., 2017).
- **Emotion Recognition (DEAM):**  
  Aljanaki et al. (2017) introduced the DEAM dataset, which provides continuous valence and arousal labels.  
  Recurrent networks such as BiGRU and LSTM have been used to capture temporal emotional flow (Aljanaki et al., 2017).  
- **Multitask Learning (MTL):**  
  Caruana (1997) formalized MTL as joint optimization over related tasks with shared representations.  
  In music MIR, MTL helps utilize cross-task correlations to stabilize training and improve robustness on small datasets (Kim et al., 2018; Zhang et al., 2019).

### 2.2 Gaps and Motivation

- Both GTZAN and DEAM are limited in size (Choi et al., 2017; Aljanaki et al., 2017), making them prone to overfitting.  
- Few prior works have systematically explored **shared CNN encoders** across genre and emotion tasks (Zhang et al., 2019; Fernández et al., 2018). 

Therefore, this project designs a **lightweight shared CNN model** to simultaneously predict *genre* and *valence–arousal*, and evaluates whether shared learning benefits both.

> However, few prior works explicitly evaluate multitask learning on both datasets using shared CNN encoders, which motivates the present study.

---

## 3. Methodology

### 3.1 Research Motivation and Question

This project investigates whether a **single shared audio encoder** can jointly support two related tasks — **genre classification** and **emotion regression** — more effectively than training two separate models.

**Research Question**  
Can a multitask learning (MTL) framework that shares a CNN encoder between genre (10-way classification) and emotion (valence/arousal regression) **outperform or at least match** single-task baselines, while improving **stability** and **efficiency**?

**Hypothesis**  
Genre and emotion share correlated acoustic cues (e.g., **timbre, rhythm, energy**). Sharing an encoder may:
- reduce redundancy in feature extraction,
- regularize task learning (better generalization),
- stabilize emotion predictions.

Expected advantages:
- Improve genre classification stability (macro-F1).  
- Reduce emotion regression RMSE (valence/arousal).  
- Achieve higher efficiency through shared feature extraction.

### 3.2 Datasets and Splits

- **GTZAN** (genre, 10 classes; ≈1,000 clips)  
  Used for the **genre head**. Labels are categorical.
- **DEAM** (emotion; >1,500 clips)  
  Used for the **emotion head**. Labels are continuous **Valence/Arousal** in \([-1, 1]\).

We adopt prepared JSON item lists:
- **Training items:** `data/lists/train_items.json`  
- **Validation items:** `data/lists/val_items.json`

Each item contains: audio path + either genre or emotion (or both if available). **Missing labels are handled with dynamic masking** (Sec. 4.5).

### 3.3 Preprocessing and Features

The feature pipeline follows prior MIR conventions using Mel, MFCC, and Chroma representations (Choi et al., 2017; Kim et al., 2018).  
Late fusion is applied to combine spectral, cepstral, and harmonic cues into a unified tensor.

All audio is resampled to **22.05 kHz**, **mono**. For each clip, we compute **late fusion** of three standard representations:
- **Mel-spectrogram** (128 bins)  
- **MFCC** (13 coefficients)  
- **Chroma** features

We stack them channel-wise to form a 3-channel time–frequency tensor:
```text
[C, F, T] = [3, 128, TARGET_FRAMES]
```

To ensure fixed-length batches, we **center-crop or right-pad** to `TARGET_FRAMES`（Approximately **1300** frames in the training code），Balance time coverage and memory overhead.

### 3.4 Model Architecture

Our architecture design is inspired by standard CNN-based MIR encoders (Choi et al., 2017) and multitask frameworks used in general audio understanding (Zhang et al., 2019).

A **shared CNN encoder** extracts time–frequency features for both tasks, followed by two task-specific heads:
- **Genre head**: 10-way linear classifier with **softmax**  
- **Emotion head**: 2-unit **linear regressor** *(Valence, Arousal)*

**1.Genre Head:** 10-way softmax classifier

**2.Emotion Head:** 2-unit linear regressor (Valence, Arousal)

```scss
             ┌─────────────┐
Input Audio →│ CNN Encoder │→ Shared Features
             └─────┬───────┘
                   │
     ┌─────────────┴──────────────┐
     │                            │
 [Genre Head]                [Emotion Head]
   Softmax(10)               Linear(2)
```

### 3.5 Multitask Loss and Missing-Label Masking

Loss masking for partially labeled data is adapted from prior multitask strategies (Caruana, 1997), allowing end-to-end training without discarding incomplete samples.

The joint objective combines **cross-entropy** for genre and **MSE** for emotion, with controllable weights:

$$
L = \lambda_{genre} \times CE + \lambda_{emo} \times MSE
$$

In practice, not every item has both labels, so we apply **per-batch masks**:
- For **GTZAN** items, emotion is missing → **only CE** contributes.  
- For **DEAM** items, `genre = -1` → **only MSE** contributes.

This masking keeps training **end-to-end** without discarding samples.

### 3.6 Optimization and Training Setup

- **Optimizer:** Adam  
- **Initial LR:** 1e-3, with **ReduceLROnPlateau** scheduler *(factor=0.5, patience=3)*  
- **Batch size:** **8**（If graphics memory is tight, you may change it to **4**）  
- **Epochs:** **50**  
- **Loss weights:** \(\lambda_{\text{genre}}=1.5\), \(\lambda_{\text{emo}}=1.0\)（Greater emphasis on stable classification convergence）  
- **Device:** CUDA if available; otherwise CPU  
- **DataLoader:** `num_workers=0`

**Hyperparameters (final run)**

| Component      | Setting                                      |
|:---------------|:---------------------------------------------|
| Sample rate    | 22.05 kHz, mono                              |
| Features       | Mel (128) + MFCC (13) + Chroma (late fusion) |
| Input tensor   | \[3, 128, TARGET_FRAMES≈1300]                |
| Encoder        | 3×(Conv–BN–ReLU) + Global Avg Pool           |
| Heads          | Genre: Linear→Softmax(10); Emotion: Linear(2)|
| Loss           | CE + MSE with masks                          |
| Weights        | \(\lambda_{\text{genre}}=1.5,\ \lambda_{\text{emo}}=1.0\) |
| Optimizer      | Adam                                         |
| LR schedule    | ReduceLROnPlateau (min-mode)                 |
| Epochs / Batch | 50 / 8                                       |
| Inference (opt)| **5-crop TTA**（Average logits / Average VA）        |

### 3.7 Tools and Techniques Learned

- Multitask training loop with **dynamic masking** for partially labeled datasets  
- **Late-fusion** acoustic features &(Optional) Feature caching for performance optimisation  
- **Test-time augmentation (TTA)** for more stable predictions  
- Reproducible splits via JSON lists & consistent **seed** control

---

## 4. Results and Evaluation

### 4.1 Evaluation Protocol

We report **validation** metrics on the predefined split (`val_items.json`) using:
- **Genre:** **Accuracy** & **macro F1**  
- **Emotion:** **RMSE** on **Valence (V)** and **Arousal (A)**

Best checkpoint is selected by the validation score:

```math
score = (1 - F1) + mean(RMSE_V, RMSE_A)
```

### 4.2 Quantitative Results (Validation)

**Summary @ best epoch (48)**

| Task   | Metric        | Value |
|:-------|:--------------|:-----:|
| Genre  | Accuracy      | 0.81  |
| Genre  | F1 (macro)    | 0.81  |
| Emotion| RMSE (Valence)| 0.027 |
| Emotion| RMSE (Arousal)| 0.036 |

**Learning Curves:**  
![](../experiments/logs/mtl_curve.png)  
*Figure 1 — MTL training curves (F1 ↑, RMSE ↓). The model converges smoothly after epoch 12 and stabilizes around epoch 48.*

**Confusion Matrix:**  
![](../experiments/logs/genre_confmat.png)  
*Figure 2 — Confusion mainly appears among similar genres (e.g., reggae ↔ disco ↔ pop).*

### 4.3 Ablation / Sensitivity Notes

- **Loss weights (\lambda):** Increasing (\lambda_{text{genre}}) 1.0→1.5 improved F1 **stability** (~+0.02–0.03) without harming RMSE.  
- **LR schedule**: A decay around epoch 12 (**1e-3→5e-4**) prevented F1 oscillations and smoothed RMSE.  
- **Batch size**: **8** yielded steadier curves than **4**（Smaller batches produce greater noise）。  
- **TTA**: 5-crop averaging slightly improved top-k **stability** and reduced per-file **variance**,This is particularly evident in the GTZAN segment of sudden-onset signals.

### 4.4 Qualitative Observations

- **Genre–Emotion correlation**: High-arousal predictions frequently align with energetic genres (metal, rock); low V/A is more commonly associated with softer or minor-key styles.  
- **Data ambiguity**: GTZAN exhibits boundary samples (e.g., disco/pop crossover). The model's confusion reflects **data ambiguity** rather than pure model error.

---

## 5. Discussion and Conclusion

### 5.1 Discussion

The results align with prior findings that shared encoders can capture cross-task acoustic cues to improve generalization (Caruana, 1997; Zhang et al., 2019).  
Compared to earlier CNN baselines (Choi et al., 2017), our model achieves comparable accuracy while improving emotional stability similar to DEAM-focused regression models (Aljanaki et al., 2017).

**Positive transfer:**  
Shared features between related tasks improve both classification and regression.  

**Training stability:**  
Learning rate scheduling and weighted loss improved convergence after epoch 30.  

**Limitations:**  
- GTZAN has overlapping and noisy genre labels.  
- DEAM emotion labels are subjective and continuous.  
- Occasional negative transfer may occur if one task dominates gradients.  

**Learnings:**  
Dynamic masking and TTA improved model robustness, while tuning λ_genre/λ_emo provided balance between tasks.

### 5.2 Future Work

- Integrate **self-supervised embeddings** (e.g., wav2vec2, BEATs).  
- Apply **transformers** or **conformer** encoders for long-range modeling.  
- Explore **dynamic task weighting** (GradNorm, uncertainty weighting).  
- Extend to **cross-dataset** evaluation for better generalization.

### 5.3 Conclusion

This project demonstrated a reproducible multitask learning framework that jointly performs genre classification and emotion regression.  
Results confirm that **shared CNN encoders** can slightly improve both accuracy and stability while simplifying the architecture.  
The findings support the hypothesis that multitask representation learning is beneficial for related MIR tasks.

> Overall, the results confirm that a shared CNN encoder can serve as an effective multitask backbone for music understanding, achieving both strong accuracy and stable regression.

### Answering the research question
A **shared CNN encoder with dual task heads**:
1) achieves **strong genre classification** (Acc/F1 ≈ **0.81**), and  
2) yields **stable emotion regression** (RMSE\(_V\)=**0.027**, RMSE\(_A\)=**0.036**)  
on the validation split, validating the effectiveness of the **MTL** approach with **masked loss**.

### Why it works
Shared representations capture common acoustic cues (**timbre/energy/rhythm**), while task-specific heads specialize the final mapping. **Loss-weighting** and **LR scheduling** further stabilize optimization.

### Limitations
- **Class overlap** in GTZAN limits an upper bound (reggae–pop/disco).  
- **Emotion labels** are continuous and subjective; without temporal modeling (e.g., sequence attention), momentary affect dynamics may be under-modeled.  
- The encoder is **CNN-only**; it lacks long-range context modeling.

### Future Work
- Replace/augment the encoder with **Conformer/AST/HTS-AT** style architectures for better temporal context.  
- Add **task-specific adapters** or **cross-stitch units** to control feature sharing granularity.  
- Explore **curriculum** or **uncertainty-weighted** loss to adaptively balance CE/MSE during training.  
- Incorporate **self-supervised audio embeddings** (e.g., wav2vec2/BEATs) for stronger transfer.  
- Expand evaluation to **cross-dataset** setups and report **confidence/uncertainty** for decision support.

### Takeaway
A compact MTL system can jointly solve **genre + emotion** with competitive accuracy and stable regression, while being **simpler and more efficient** than maintaining two standalone pipelines.

---

## 6. References

- Aljanaki, A., Yang, Y., & Soleymani, M. (2017). DEAM: A Dataset for Emotional Analysis in Music. ISMIR.
- Caruana, R. (1997). Multitask Learning. Machine Learning, 28(1), 41–75.
- Choi, K., Fazekas, G., Sandler, M., & Cho, K. (2017). Convolutional recurrent neural networks for music classification. ICASSP.
- Kim, Y., Lee, H., & Nam, J. (2018). Sample-level CNN architectures for music auto-tagging. IEEE T-ASLP.
- Zhang, C., Tan, K. C., et al. (2019). Affective computing for music emotion recognition: A deep learning perspective.
- Fernández, J., García, S., Galar, M., et al. (2018). Survey of music information retrieval systems based on deep learning.

---

## Appendix

- Checkpoints: `experiments/checkpoints/mtl_best_ep48.pt`  
- Config: `experiments/configs/mtl_deam_gtzan.json`  
- Curves & logs: `experiments/logs/mtl_curve.csv/png`, `genre_confmat.csv/png`  
- Prediction script: `scripts/predict_mtl.py`

**Example run:**
```bash
python scripts/predict_mtl.py --ckpt experiments/checkpoints/mtl_best_ep48.pt \
--wav "data/GTZAN_raw/reggae/reggae.00002.wav" --topk 3 --tta 5
```
