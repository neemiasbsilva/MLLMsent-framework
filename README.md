# PerceptSent-LLM Approach

## Overview

PerceptSent-LLM is a comprehensive research framework for sentiment analysis using Large Language Models (LLMs) and modern NLP techniques. The framework provides tools for training, evaluating, and comparing various transformer-based models on perceptual sentiment datasets.

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
   git clone <repo-url>
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
├── data/                # Datasets and model outputs
│   ├── gpt4-openai-classify/    # GPT-4 classifications
│   ├── minigpt4-classify/       # MiniGPT-4 classifications
│   └── deepseek-classify/       # DeepSeek VL-2 classifications
├── models/              # Model architectures and utilities
├── utils/               # Helper functions and tools
├── experiments/         # Experiment configurations and results
├── experiments-not-finetuning/  # Non-fine-tuning experiments
├── experiments-swin/    # Swin Transformer experiments
├── checkpoints/         # Model checkpoints
├── scripts/            # Training and evaluation scripts
├── notebooks/          # Analysis and prototyping notebooks
├── reports/            # Results and visualizations
└── requirements.txt    # Project dependencies
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
   ```bash
   ./run.sh
   ```

### Notebooks
- `fine-tuning-llm-qlora.ipynb`: QLoRA fine-tuning
- `zero-shot-bart-large-mnli.ipynb`: Zero-shot classification
- `plot-results.ipynb`: Results visualization

## Results

### Performance Comparison

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

## Contributing

We welcome contributions! Please feel free to submit a Pull Request.

## Citation

If you use this repository in your research, please cite:
```
[Add citation information here]
```

## License

[Add license information here]

## Contact

For questions or suggestions, please open an issue in the repository.