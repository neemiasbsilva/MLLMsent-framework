# PerceptSent-LLM Approach

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-%23FF6F00.svg?style=for-the-badge&logo=huggingface&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%23FF6F00.svg?style=for-the-badge&logo=huggingface&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![CUDA](https://img.shields.io/badge/CUDA-%23076FC1.svg?style=for-the-badge&logo=nvidia&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

## Overview

PerceptSent-LLM is a comprehensive research framework for investigating sentiment reasoning capabilities of MultiModal Large Language Models (MLLMs). This repository provides a complete implementation for sentiment analysis from visual content, addressing the challenging problem of understanding how images communicate sentiment through complex, scene-level semantics.

The framework implements three main approaches for sentiment analysis:

1. **Direct sentiment classification** from images using MLLMs
2. **Sentiment analysis on MLLM-generated captions** using pre-trained LLMs with only the final classification layer trained for sentiment polarity
3. **Full fine-tuning** of the LLMs on sentiment-labeled captions

This repository includes implementations of various transformer architectures (ModernBERT, BART, LLaMA, DistilBERT, Swin Transformer) and provides tools for both fine-tuning and non-fine-tuning experiments. The framework has demonstrated state-of-the-art performance, outperforming CNN- and Transformer-based baselines by up to 15% across different sentiment polarity categories.

### Key Features
- End-to-end pipeline for sentiment analysis with LLMs
- Support for multiple transformer architectures and training strategies
- Fine-tuning with qLORA and quantization
- Zero-shot and few-shot evaluation capabilities
- Comprehensive experiment tracking and reproducibility
- Modular codebase for easy extension

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/neemiasbsilva/PerceptSent-LLM-approach.git
   cd PerceptSent-LLM-approach
   ```

2. **Create checkpoints directory and download model weights:**
   ```bash
   mkdir checkpoints
   ```
   
   Download pre-trained model weights from:
   [Google Drive - PerceptSent-LLM Model Weights](https://drive.google.com/drive/u/0/folders/1eumPYLgpk7Gr71lG0j6MtgTpnfbhiBr9)
   
   The available checkpoints include:
   - **OpenAI ModernBERT** variants (p3, p5, sigma5)
   - **DeepSeek ModernBERT** variants (p3, p5, sigma5) 
   - **OpenAI BART** variants (p5, sigma5)
   - **DeepSeek BART** variants (p5, sigma5)
   
   Extract the downloaded `.pt.gz` files to the `checkpoints/` directory:
   ```bash
   gunzip checkpoints/*.pt.gz
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

```
PerceptSent-LLM-approach/
├── data/                           # Datasets and model outputs
│   ├── gpt4-openai-classify/       # GPT-4 classifications
│   ├── gpt4-openai-only/           # GPT-4 only outputs
│   ├── minigpt4-classify/          # MiniGPT-4 classifications
│   ├── deepseek/                   # DeepSeek VL-2 outputs
│   ├── percept_dataset/            # Perceptual sentiment dataset
│   ├── twiter/                     # Twitter dataset
│   ├── raw/                        # Raw data files
│   ├── train/                      # Training data splits
│   ├── test/                       # Test data splits
│   └── validation/                 # Validation data splits
├── models/                         # Model architectures and utilities
├── utils/                          # Helper functions and tools
├── experiments/                    # Main experiment configurations and results
├── experiments-finetuning/         # Fine-tuning experiment results
├── experiments-not-finetuning/     # Non-fine-tuning experiment results
├── experiments-swin/               # Swin Transformer experiments
├── experiments-twitter/            # Twitter-specific experiments
├── checkpoints/                    # Model checkpoints
├── scripts/                        # Training and evaluation scripts
├── notebooks/                      # Analysis and prototyping notebooks
├── reports/                        # Results and visualizations
├── textaugment/                    # Text augmentation utilities
├── envmodernbert/                  # ModernBERT environment
├── run-*.sh                        # Execution scripts for different experiments
├── requirements.txt                # Project dependencies
├── pyproject.toml                  # Python project configuration
└── uv.lock                         # Dependency lock file
```

## Data Structure

### Model Classifications

#### GPT-4 OpenAI Classifications
Located in `/data/gpt4-openai-classify/`
- Alpha values: 3, 4, 5
- Prompt types:
  - p2neg: Negative prompt type 2
  - p2plus: Positive prompt type 2
  - p3: Prompt type 3
  - p5: Prompt type 5

#### MiniGPT-4 Classifications
Located in `/data/minigpt4-classify/`
- Sigma values: 3, 4, 5
- Prompt types: p2neg, p2plus, p3, p5

#### DeepSeek VL-2 Classifications
Located in `/data/deepseek-classify/`
- Sigma values: 3, 4, 5
- Prompt types: p2neg, p2plus, p3, p5

## Experimental Setup

### Model Variants

#### ModernBERT Experiments
- OpenAI and DeepSeek variants
- Alpha values: 3, 4, 5
- Prompt types: p2neg, p2plus, p3, p5

#### BART Experiments
- Zero-shot BART
- DeepSeek BART variants
- Alpha values: 3, 4, 5
- Prompt types: p2neg, p2plus, p3, p5

#### LLaMA3 QLoRA Experiments
- OpenAI and DeepSeek variants
- Alpha values: 3, 4, 5
- Prompt types: p2neg, p2plus, p3, p5

#### DistilBERT Experiments
- OpenAI and DeepSeek variants
- Alpha values: 3, 4, 5
- Prompt types: p2neg, p2plus, p3, p5

#### Swin Transformer Experiments
- OpenAI variants
- Alpha values: 3, 5
- Prompt types: p3, p5

### Additional Experiments
- VADER sentiment analysis
- Text augmentation studies

## Usage

### Training & Evaluation

1. **Training a model:**
   ```bash
   python scripts/train.py --config <path-to-config.yaml>
   ```

2. **Evaluating a model:**
   ```bash
   python scripts/evaluate.py
   ```

3. **Running experiments:**

   **Fine-tuning experiments:**
   ```bash
   # BART fine-tuning
   ./run-finetuning-bart.sh
   
   # ModernBERT fine-tuning
   ./run-finetuning-modern-bert.sh
   
   # LLaMA fine-tuning
   ./run-finetuning-llama.sh
   ```

   **Non-fine-tuning experiments:**
   ```bash
   # BART non-fine-tuning
   ./run-not-finetuning-bart.sh
   
   # ModernBERT non-fine-tuning
   ./run-not-finetuning-modern-bert.sh
   ```

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## Citation

If you use this repository in your research, please cite:
```
[Add citation information here]
```
