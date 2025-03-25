# NLP-Project-CounterSpeech Baselines using DialoGPT

## Introduction
This project fine-tunes Microsoft's DialoGPT-small model to generate counterspeech responses to hate speech. Using a custom dataset of hate speech and counterspeech pairs, the system leverages the Hugging Face `transformers` library, PyTorch, and various evaluation metrics (ROUGE, BERTScore) to train and assess the model's performance. The repository includes two main Jupyter notebooks: one for training the model and another for inference and evaluation. This README provides instructions for running the evaluation notebook (`eval.ipynb`), assumed to be located in a `Baselines` directory.

## Prerequisites
Before running the project, ensure the following are installed:
- **Python**: Version 3.10 or higher
- **Jupyter Notebook**: For running `.ipynb` files
- **GPU (Optional)**: NVIDIA GPU with CUDA support for faster execution
- **pip**: Python package manager

## Installation
The project dependencies are listed in `requirements.txt`. See the Execution Instructions below for installation steps.

## Execution Instructions
To run the evaluation notebook, follow these steps:

```bash
cd Baselines
pip install -r requirements.txt
jupyter notebook eval.ipynb
```

## Note

Loading the model and performing decoding operations require a significant amount of RAM. Ensure your system has sufficient memory available to avoid performance issues or crashes during execution.