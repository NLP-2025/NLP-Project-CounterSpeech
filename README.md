# NLP Project: CounterSpeech Baselines

This repository contains implementation of various models for generating counter-speech responses to harmful content. The project compares different architectural approaches against baseline models.

## Abstract

We explore intent-specific counterspeech generation to tackle hate speech online. Using the IntentCONAN v2 dataset—with 9,532 training examples balanced across four rhetorical intents (Informative, Denouncing, Questioning, and Positive)—we propose a modular framework with a shared HateBERT encoder and intent-specific BART decoders. We investigate three fusion mechanisms (Linear, Shared, and Cross Attention) to combine hate speech embeddings with intent representations. For evaluation, we introduce DialoRank, a zero-shot DialoGPT method that ranks responses by intent relevance. Results show our intent-aware models outperform DialoGPT and GPS baselines across lexical and semantic metrics, with SharedFusion achieving the best performance (ROUGE-1: 0.251, METEOR: 0.158, BERTScore F1: 0.871). Our findings highlight the effectiveness of intent conditioning for generating contextually appropriate and rhetorically impactful counterspeech.

## Models Implemented

### Baselines
- **DialoGPT**: A standard dialogue generation model
- **GPS**: Generative Pre-trained Speech model

### QUADRA Architecture Variants
- **Linear Fusion**: Simple linear combination of features
- **Cross Attention**: Cross-attention mechanism between features
- **Shared Fusion**: Shared representation learning

## How to Run

### 1. DialoGPT Baseline
```bash
cd Baselines/DialoGPT
run eval_submission.ipynb
```

### 2. GPS
Run the GPS baseline evaluation notebooks in the Baselines/GPS directory.

### 3. QUADRA Architectures

#### A. Linear Fusion
```bash
cd Main_architectures/Linear\ Fusion
run eval_linear_fusion.ipynb
```

#### B. Cross Fusion
```bash
cd Main_architectures/Cross\ Attention
run eval_cross_attention.ipynb
```

#### C. Shared Fusion
```bash
cd Main_architectures/Shared\ Fusion
run eval_shared_fusion.ipynb
```

## Results

### Text Generation Metrics

| Model | R1 | R2 | RL | M |
|-------|-------|-------|-------|-------|
| LinearFusion | 0.250 | 0.064 | 0.175 | 0.154 |
| SharedFusion | 0.251 | 0.065 | 0.176 | 0.158 |
| CrossFusion | 0.242 | 0.061 | 0.171 | 0.152 |
| DialoGPT | 0.130 | 0.003 | 0.105 | 0.040 |
| GPS | 0.176 | 0.030 | 0.132 | 0.116 |

### Semantic Similarity and Classification Metrics

| Model | BS(P) | BS(R) | BS(F1) | CA |
|-------|-------|-------|--------|-------|
| LinearFusion | 0.869 | 0.871 | 0.870 | 0.752 |
| SharedFusion | 0.871 | 0.870 | 0.871 | 0.751 |
| CrossFusion | 0.870 | 0.869 | 0.870 | 0.752 |
| DialoGPT | 0.791 | 0.808 | 0.799 | 0.681 |
| GPS | 0.240 | 0.121 | 0.180 | 0.754 |

## QUARC Model Implementation

The implementation of the **QUARC** model is available at the official GitHub repository:

🔗 [https://github.com/LCS2-IIITD/quarc-counterspeech](https://github.com/LCS2-IIITD/quarc-counterspeech)

This model is introduced in the paper:

**Counterspeeches up my sleeve! Intent Distribution Learning and Persistent Fusion for Intent-Conditioned Counterspeech Generation**  
*Rishabh Gupta, Shaily Desai, Manvi Goel, Anil Bandhakavi, Tanmoy Chakraborty, and Md. Shad Akhtar*  
📄 [arXiv:2305.13776](https://arxiv.org/abs/2305.13776)


## 🧠 Model Weights

The pre-trained weights for all the models used in this project — including **DialoGPT FineTuned**, **Linear Fusion**, **Shared Fusion**, and **Cross Attention** — are available for download.

📁 **Download here**: [Model Weights on Google Drive](https://drive.google.com/drive/folders/1rFA1at5oYa7uyDG3KWniPgkl2ZZlizVD?usp=sharing)
