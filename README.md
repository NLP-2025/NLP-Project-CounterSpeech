# NLP Project: CounterSpeech Baselines

This repository contains implementation of various models for generating counter-speech responses to harmful content. The project compares different architectural approaches against baseline models.

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