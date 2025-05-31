# PerceptSent-LLM-approach

## Overview

**PerceptSent-LLM-approach** is a research framework for sentiment analysis using Large Language Models (LLMs) and modern NLP techniques. It provides tools for training, evaluating, and comparing various transformer-based models (including GPT-4o, BART-LARGE-MNLI, DISTIL-BERT, LLAMA-3, ModernBERT, and more) on perceptual sentiment datasets. The repository supports zero-shot, few-shot, and fine-tuning paradigms, and includes scripts, experiment configs, and Jupyter notebooks for reproducibility.

## Features
- End-to-end pipeline for sentiment analysis with LLMs
- Support for multiple transformer architectures and training strategies
- Fine-tuning with qLORA and quantization (LLAMA-3, BART, DISTIL-BERT, ModernBERT, etc.)
- Zero-shot and few-shot evaluation (BART-LARGE-MNLI, GPT-4o, etc.)
- Experiment tracking and reproducibility
- Comprehensive results and visualizations
- Modular codebase for easy extension

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd PerceptSent-LLM-approach
   ```
2. **Install dependencies:**
   It is recommended to use a virtual environment (e.g., `venv` or `conda`).
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training & Evaluation
- **Train a model:**
  ```bash
  python scripts/train.py --config <path-to-config.yaml>
  ```
  Example configs are in `experiments/` and `experiments-not-finetuning/`.

- **Evaluate a model:**
  ```bash
  python scripts/evaluate.py
  ```

- **Run all experiments:**
  Use the provided `run.sh` to execute a batch of experiments for different models and settings.

### Notebooks
- Example and exploratory notebooks are available in the `notebooks/` directory:
  - `fine-tuning-llm-qlora.ipynb`: Fine-tuning LLMs with QLoRA
  - `zero-shot-bart-large-mnli.ipynb`: Zero-shot classification with BART-LARGE-MNLI
  - `plot-results.ipynb`: Visualization of results

### Data
- Place your datasets in the `data/` directory. The structure supports multiple splits and sources.

## Results

### F1-Score Comparison Table

| Type of dominance | Threshold (σ) | Classification Problem | Percept Sent Paper ResNet | GPT4-o mini + Vader | GPT4-o mini + Zero-shot BART-LARGE-MNLI | GPT4-o mini | GPT4-o mini + Fine-Tuning DISTIL-BERT | GPT4-o mini + Fine-Tuning BART-LARGE-MNLI | GPT4-o mini + Fine-Tuning LLAMA-3 (qLORA) | GPT4-o mini + Fine-Tuning ModernBERT |
|-------------------|---------------|-----------------------|--------------------------|--------------------|----------------------------------------|-------------|---------------------------------------|-------------------------------------------|--------------------------------------------|-------------------------------------|
| Simple            | σ = 3         | P5 (C=5)              | 45.00% [± 0.034]         | -                  | 48.25% [± 0.043]                       | 44.58% [± 0.031] | 56.74% [± 0.042]                  | 56.35% [± 0.039]                        | 59.45% [± 0.037]                         | 58.47% [± 0.04]                    |
|                   |               | P3 (C=3)              | 61.00% [± 0.053]         | 5.27% [± 0.009]    | 67.64% [± 0.018]                       | 61.28% [± 0.029] | 76.44% [± 0.015]                  | 75.47% [± 0.012]                        | 77.53% [± 0.018]                         | 75.42% [± 0.02]                    |
|                   |               | P2+ (C=2)             | 64.40% [± 0.062]         | 51.54% [± 0.019]   | 78.15% [± 0.013]                       | 72.63% [± 0.011] | 82.11% [± 0.006]                  | 82.33% [± 0.005]                        | 83.36% [± 0.014]                         | 83.05% [± 0.01]                    |
|                   |               | P2- (C=2)             | 74.80% [± 0.050]         | 45.58% [± 0.023]   | 73.44% [± 0.019]                       | 82.84% [± 0.009] | 85.75% [± 0.014]                  | 82.53% [± 0.027]                        | 85.32% [± 0.009]                         | 85.03% [± 0.01]                    |
| Qualified         | σ = 4         | P5 (C=5)              | 48.20% [± 0.089]         | -                  | 62.28% [± 0.029]                       | 50.06% [± 0.026] | 71.32% [± 0.028]                  | 69.75% [± 0.009]                        | 69.02% [± 0.023]                         | 72.26% [± 0.04]                    |
|                   |               | P3 (C=3)              | 68.00% [± 0.052]         | 1.95% [± 0.003]    | 77.05% [± 0.016]                       | 70.53% [± 0.022] | 88.73% [± 0.013]                  | 87.79% [± 0.015]                        | 87.43% [± 0.016]                         | 88.16% [± 0.02]                    |
|                   |               | P2+ (C=2)             | 76.60% [± 0.062]         | 56.21% [± 0.016]   | 84.83% [± 0.024]                       | 82.51% [± 0.023] | 90.48% [± 0.020]                  | 90.33% [± 0.010]                        | 90.94% [± 0.006]                         | 91.22% [± 0.01]                    |
|                   |               | P2- (C=2)             | 84.00% [± 0.035]         | 47.86% [± 0.025]   | 78.81% [± 0.019]                       | 88.51% [± 0.012] | 92.04% [± 0.017]                  | 91.43% [± 0.018]                        | 91.63% [0.001]                            | 94.26% [± 0.01]                    |
| Absolute          | σ = 5         | P5 (C=5)              | 51.20% [± 0.063]         | -                  | 78.49% [± 0.066]                       | 75.79% [± 0.047] | 83.17% [± 0.066]                  | 82.87% [± 0.089]                        | 83.68% [± 0.054]                         | 84.44% [± 0.03]                    |
|                   |               | P3 (C=3)              | 69.00% [± 0.047]         | 0.40% [± 0.002]    | 84.67% [± 0.029]                       | 87.68% [± 0.018] | 95.53% [± 0.011]                  | 95.63% [± 0.017]                        | 92.70% [± 0.019]                         | 96.02% [± 0.01]                    |
|                   |               | P2+ (C=2)             | 81.20% [± 0.054]         | 62.28% [± 0.035]   | 89.61% [± 0.018]                       | 85.51% [± 0.016] | 95.42% [± 0.013]                  | 95.47% [± 0.009]                        | 94.60% [± 0.006]                         | 95.90% [± 0.01]                    |
|                   |               | P2- (C=2)             | 88.60% [± 0.011]         | 48.81% [± 0.050]   | 84.56% [± 0.022]                       | 93.77% [± 0.018] | 96.44% [± 0.012]                  | 96.33% [± 0.009]                        | 96.10% [0.0102]                            | 96.24% [± 0.01]                    |


## Project Structure

```
PerceptSent-LLM-approach/
├── data/                # Datasets (train/validation/test, raw, processed)
├── models/              # Model definitions (architectures, utils)
├── utils/               # Utility scripts (data loading, visualization)
├── experiments/         # Experiment configs and logs
├── experiments-not-finetuning/ # Additional experiment configs
├── checkpoints/         # Saved model checkpoints
├── scripts/             # Training, evaluation scripts
├── notebooks/           # Jupyter notebooks for analysis and prototyping
├── reports/             # Results, plots, and CSVs
├── requirements.txt     # Python dependencies
├── run.sh               # Batch script for running experiments
└── README.md            # Project documentation
```

## Citation
If you use this repository or its results in your research, please cite appropriately (add your citation here).

## Contact & Contributions
For questions, suggestions, or contributions, please open an issue or pull request.

---