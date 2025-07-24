# PerceptSent-LLM Approach

![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Transformers](https://img.shields.io/badge/Transformers-%23FF6F00.svg?style=for-the-badge&logo=huggingface&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-%23FF6F00.svg?style=for-the-badge&logo=huggingface&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![CUDA](https://img.shields.io/badge/CUDA-%23076FC1.svg?style=for-the-badge&logo=nvidia&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)

---

## Overview

**PerceptSent-LLM** is a research framework for investigating sentiment reasoning in MultiModal Large Language Models (MLLMs). It provides end-to-end tools for sentiment analysis from visual content, focusing on how images communicate sentiment through complex, scene-level semantics.

- **Direct sentiment classification** from images using MLLMs
- **Sentiment analysis on MLLM-generated captions** using pre-trained LLMs (with only the final classification layer trained)
- **Full fine-tuning** of LLMs on sentiment-labeled captions

The framework supports multiple transformer architectures (ModernBERT, BART, LLaMA, DistilBERT, Swin Transformer) and both fine-tuning and non-fine-tuning experiments. It achieves state-of-the-art performance, outperforming CNN/Transformer baselines by up to 15% across sentiment categories.

### Key Features
- End-to-end pipeline for sentiment analysis with LLMs
- Support for multiple transformer architectures and training strategies
- Fine-tuning with qLORA and quantization
- Zero-shot and few-shot evaluation
- Comprehensive experiment tracking and reproducibility
- Modular, extensible codebase

---

## Quickstart

```bash
# Clone the repository
$ git clone https://github.com/neemiasbsilva/PerceptSent-LLM-approach.git
$ cd PerceptSent-LLM-approach

# Create checkpoints directory and download model weights
$ mkdir checkpoints
# Download weights from:
# https://drive.google.com/drive/u/0/folders/1eumPYLgpk7Gr71lG0j6MtgTpnfbhiBr9
# Extract with:
$ gunzip checkpoints/*.pt.gz

# Install dependencies (Python >=3.10 required)
$ pip install -r requirements.txt
# or, for modern Python projects:
$ pip install .
```

---

## Project Structure

```
PerceptSent-LLM-approach/
├── data/                 # Datasets and model outputs
├── models/               # Model architectures and utilities
├── utils/                # Helper functions and tools
├── experiments-finetuning/      # Fine-tuning experiment configs/results
├── experiments-not-finetuning/  # Non-fine-tuning experiment configs/results
├── experiments-swin/     # Swin Transformer experiments
├── experiments-twitter/  # Twitter-specific experiments
├── checkpoints/          # Model checkpoints
├── scripts/              # Training, evaluation, and inference scripts
├── notebooks/            # Analysis and prototyping notebooks
├── reports/              # Results and visualizations
├── textaugment/          # Text augmentation utilities
├── envmodernbert/        # ModernBERT environment
├── run-*.sh              # Shell scripts for experiments
├── requirements.txt      # Python dependencies
├── pyproject.toml        # Python project config (PEP 621)
└── uv.lock               # Dependency lock file
```

---

## Configuration

Experiments are configured via YAML files (see `experiments-finetuning/` and `experiments-not-finetuning/`). Example config:

```yaml
experiment_name: "Experiment using LLama3 Finetuning with QlORA"
learning_rate: 1e-5
batch_size: 4
epochs: 100
model_path: "nvidia/Llama3-ChatQA-1.5-8B"
model_name: "llama-qlora"
max_len: 1024
log_dir: "experiments-finetuning/llama3-qlora-p3-alpha3/logs"
checkpoint_dir: "checkpoints"
```

- **experiment_name**: Name of the experiment
- **learning_rate**: Learning rate for training
- **batch_size**: Batch size
- **epochs**: Number of epochs
- **model_path**: HuggingFace model path or identifier
- **model_name**: Model type (e.g., "llama-qlora", "modern-bert", "bart", "distil-bert")
- **max_len**: Max sequence length
- **log_dir**: Directory for logs
- **checkpoint_dir**: Directory for saving checkpoints

---

## Training & Evaluation

### Training

```bash
python scripts/train_gpu0.py --config <path-to-config.yaml>
# or
python scripts/train_gpu1.py --config <path-to-config.yaml>
# or (for Swin Transformer)
python scripts/swin_train.py --config <path-to-config.yaml>
```

### Evaluation

```bash
python scripts/evaluate.py --config <path-to-config.yaml>
```

### Running Experiments (Shell Scripts)

```bash
# Fine-tuning
./run-finetuning-bart.sh
./run-finetuning-modern-bert.sh
./run-finetuning-llama.sh

# Non-fine-tuning
./run-not-finetune-bart.sh
./run-not-finetune-modern-bert.sh
```

---

## Inference

See [`scripts/README_inference.md`](scripts/README_inference.md) for full details.

**Quick Start:**

```bash
# List available checkpoints
python scripts/run_inference.py --list

# Run inference (recommended)
python scripts/run_inference.py \
    --checkpoint checkpoints/best_checkpoint_gpt4-openai-classify_bart_p5_sigma3_finetuned.pt \
    --input your_data.csv \
    --output predictions.csv

# Or use the main script directly
python scripts/inference.py \
    --model_name bart \
    --checkpoint_path checkpoints/best_checkpoint_gpt4-openai-classify_bart_p5_sigma3_finetuned.pt \
    --model_path facebook/bart-large-mnli \
    --input_file your_data.csv \
    --output_file predictions.csv \
    --num_classes 5 \
    --batch_size 32 \
    --max_len 512
```

- Input CSV must have a `text` column (or specify with `--text_column`)
- Output CSV will have a new `prediction` column
- See the [inference README](scripts/README_inference.md) for model-specific details and troubleshooting

---

## Data Structure

- `data/` contains all datasets and model outputs, including:
  - `gpt4-openai-classify/`, `minigpt4-classify/`, `deepseek/`, etc.
  - `percept_dataset/`, `twiter/`, `raw/`, `train/`, `test/`, `validation/`

---

## Notebooks

- Prototyping, analysis, and visualization notebooks are in `notebooks/`.
- Example: `plot-results.ipynb`, `fine-tuning-llm-qlora.ipynb`, `vader.ipynb`, etc.

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request. For major changes, please open an issue first to discuss what you would like to change.

---

## Citation

If you use this repository in your research, please cite:

```
[Add citation information here]
```

---

## License

[MIT License](LICENSE)

---
