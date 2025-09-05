# ELEC5305 Project Proposal  

## 1. Project Title  
**Joint Music Emotion Recognition and Genre Classification via Multi-Task Learning with Late-Fusion Features**

---

## 2. Student Information  
- **Full Name**: Kejia Huanng  
- **Student ID (SID)**: 540373974  
- **GitHub Username**: Khua0419  
- **GitHub Project Link**: [GitHub Repository](https://github.com/Khua0419/music-multitask-genre-emotion/tree/main)  

---

## 3. Project Overview  
This project explores whether multi-task learning with late-fusion features can improve both music genre classification and music emotion recognition compared to single-task baselines. The problem addressed is that existing systems typically treat genre and emotion as independent tasks, which limits their ability to capture the shared representations underlying musical perception.  

The project is important because genre and emotion are two of the most commonly used descriptors in music information retrieval (MIR). An integrated system that leverages their correlation could enable better music recommendation, emotion-aware applications, and affective computing.  

Our proposed solution is to design a multi-task learning framework with a shared feature extractor and two task-specific output heads (genre and emotion). We will further incorporate late fusion of multiple acoustic features (Mel-spectrograms, MFCC, and Chroma) to investigate whether combining complementary representations can improve robustness and accuracy.  

---

## 4. Background and Motivation  
Previous studies in MIR have shown that both genre classification (Tzanetakis & Cook, 2002) and music emotion recognition (Aljanaki et al., 2017) are feasible using deep learning models. However, most research treats these problems separately. In reality, genre and emotion are correlated but not identical descriptors of music.  

Recent advances in multi-task learning and feature fusion (Choi et al., 2017; Zhang et al., 2022) suggest that leveraging shared representations across tasks can lead to improved generalization. This motivates us to investigate a joint framework that integrates both genre and emotion recognition.  

I chose this topic because it combines my interest in audio signal processing, machine learning, and affective computing, and it builds directly on the starter codes provided in the course (speech recognition, keyword spotting, late fusion, and speech emotion recognition).  

---

## 5. Proposed Methodology  
- **Tools and Platforms**: Python, PyTorch, Librosa, Scikit-learn  
- **Signal Processing Techniques**: Mel-spectrogram, MFCC, Chroma, SpecAugment  
- **Models**: Convolutional Recurrent Neural Network (CRNN) as the baseline backbone; Transformer encoder (optional, advanced)  
- **Multi-task Framework**: Shared backbone with two heads  
  - Head 1: Genre classification (softmax)  
  - Head 2: Emotion classification (softmax) or AV regression (MSE/CCC loss)  
- **Fusion Strategy**: Late fusion at feature level (concatenation) and model output level (weighted ensemble)  
- **Datasets**:  
  - GTZAN: 1,000 audio clips across 10 genres (30s each)  
  - DEAM / EmoMusic: ~1,500 songs annotated with categorical or continuous arousal–valence emotion labels  

---

## 6. Expected Outcomes  
- A working prototype (PyTorch code) hosted on GitHub with clear documentation.  
- A multi-task model that performs both genre classification and emotion recognition.  
- Performance metrics:  
  - **Genre**: Accuracy, Macro-F1, Confusion Matrix  
  - **Emotion (categorical)**: Macro-F1, ROC-AUC  
  - **Emotion (AV regression)**: RMSE, Concordance Correlation Coefficient (CCC)  
- A 10-page final report in academic style.  
- A short demo video showing the system applied to real music clips.  

---

## 7. Timeline (Weeks 6–13)  

| Week | Task |
|------|------|
| 6–7  | Literature review, dataset collection (GTZAN, DEAM), and feature extraction pipeline setup |
| 8–9  | Implement single-task baselines (genre CNN/CRNN; emotion CNN/CRNN) and evaluate performance |
| 10–11| Develop and test the multi-task framework; experiment with different loss weights (λ) |
| 12   | Implement late-fusion experiments (Mel, MFCC, Chroma) and compare with baselines |
| 13   | Final report writing, GitHub documentation, and demo video preparation |

---

## 8. References  
- Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. *IEEE Transactions on Speech and Audio Processing, 10*(5), 293–302. https://doi.org/10.1109/TSA.2002.800560  
- Aljanaki, A., Yang, Y.-H., & Soleymani, M. (2017). Developing a benchmark for emotional analysis of music. *PLoS ONE, 12*(3), e0173392. https://doi.org/10.1371/journal.pone.0173392  
- Choi, K., Fazekas, G., Sandler, M., & Cho, K. (2017). Convolutional recurrent neural networks for music classification. In *ICASSP 2017* (pp. 2392–2396). IEEE. https://doi.org/10.1109/ICASSP.2017.7952585  
- Zhang, S., Xu, H., & Yang, Y.-H. (2022). Music emotion recognition: A survey of datasets, features, and methods. *IEEE Transactions on Affective Computing, 13*(4), 2019–2036. https://doi.org/10.1109/TAFFC.2020.2981442  

---

