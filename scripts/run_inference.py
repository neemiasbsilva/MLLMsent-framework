#!/usr/bin/env python3
"""
Helper script for running inference with trained models.
This script automatically determines the correct parameters based on checkpoint naming conventions.
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path

def parse_checkpoint_name(checkpoint_path):
    """
    Parse checkpoint name to extract model information.
    
    Expected format: best_checkpoint_{dataset_type}_{model}_{experiment_group}_sigma{alpha_version}_{finetuning}.pt
    
    Examples:
    - best_checkpoint_gpt4-openai-classify_bart_p5_sigma3_finetuned.pt
    - best_checkpoint_minigpt4-classify_modernbert_p3_sigma5_finetuned.pt
    - best_checkpoint_deepseek_modernbert_p5_sigma5_finetuned.pt
    """
    checkpoint_name = os.path.basename(checkpoint_path)
    
    # Remove .pt extension
    if checkpoint_name.endswith('.pt'):
        checkpoint_name = checkpoint_name[:-3]
    
    # Split by underscores
    parts = checkpoint_name.split('_')
    
    if len(parts) < 6:
        raise ValueError(f"Invalid checkpoint name format: {checkpoint_name}")
    
    # Extract components
    dataset_type = parts[2]  # gpt4-openai-classify, minigpt4-classify, deepseek
    model = parts[3]  # bart, modernbert
    experiment_group = parts[4]  # p5, p3, p2plus, p2neg
    alpha_version = parts[5].replace('sigma', '')  # 3, 4, 5
    finetuning = parts[6] if len(parts) > 6 else "finetuned"  # finetuned, not_finetuned
    
    return {
        'dataset_type': dataset_type,
        'model': model,
        'experiment_group': experiment_group,
        'alpha_version': alpha_version,
        'finetuning': finetuning
    }

def get_model_path(model_name):
    """Get the base model path for the given model name."""
    model_paths = {
        'bart': 'facebook/bart-large-mnli',
        'modernbert': 'microsoft/DialoGPT-medium',  # This should be updated with the actual ModernBERT path
        'llama': 'meta-llama/Llama-2-7b-chat-hf'  # This should be updated with the actual Llama path
    }
    
    if model_name not in model_paths:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model_paths[model_name]

def get_num_classes(experiment_group):
    """Get the number of classes based on experiment group."""
    class_mapping = {
        'p5': 5,  # 5 classes: Negative, SlightlyNegative, Neutral, SlightlyPositive, Positive
        'p3': 3,  # 3 classes: Negative, Neutral, Positive
        'p2plus': 2,  # 2 classes: Negative, Neutral
        'p2neg': 2   # 2 classes: Positive, Neutral
    }
    
    if experiment_group not in class_mapping:
        raise ValueError(f"Unknown experiment group: {experiment_group}")
    
    return class_mapping[experiment_group]

def run_inference(checkpoint_path, input_file, output_file, batch_size=32, max_len=512):
    """
    Run inference using the specified checkpoint.
    
    Args:
        checkpoint_path (str): Path to the model checkpoint
        input_file (str): Path to input CSV file
        output_file (str): Path to save predictions
        batch_size (int): Batch size for inference
        max_len (int): Maximum sequence length
    """
    
    # Parse checkpoint name
    checkpoint_info = parse_checkpoint_name(checkpoint_path)
    
    # Determine model name for inference script
    model_name_mapping = {
        'bart': 'bart',
        'modernbert': 'modern-bert',
        'llama': 'llama'
    }
    
    model_name = model_name_mapping.get(checkpoint_info['model'])
    if model_name is None:
        raise ValueError(f"Unknown model in checkpoint: {checkpoint_info['model']}")
    
    # Get model path
    model_path = get_model_path(checkpoint_info['model'])
    
    # Get number of classes
    num_classes = get_num_classes(checkpoint_info['experiment_group'])
    
    # Build inference command
    cmd = [
        sys.executable, 'scripts/inference.py',
        '--model_name', model_name,
        '--checkpoint_path', checkpoint_path,
        '--model_path', model_path,
        '--input_file', input_file,
        '--output_file', output_file,
        '--num_classes', str(num_classes),
        '--batch_size', str(batch_size),
        '--max_len', str(max_len)
    ]
    
    # Add experiment group for Llama
    if model_name == 'llama':
        cmd.extend(['--experiment_group', checkpoint_info['experiment_group']])
    
    print(f"Running inference with command:")
    print(' '.join(cmd))
    print()
    
    # Run the inference
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("Inference completed successfully!")
        print(result.stdout)
    else:
        print("Inference failed!")
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        sys.exit(1)

def list_available_checkpoints():
    """List all available checkpoints in the checkpoints directory."""
    checkpoints_dir = Path("checkpoints")
    
    if not checkpoints_dir.exists():
        print("Checkpoints directory not found.")
        return
    
    print("Available checkpoints:")
    print("-" * 80)
    
    for checkpoint_file in checkpoints_dir.glob("*.pt"):
        try:
            info = parse_checkpoint_name(str(checkpoint_file))
            print(f"File: {checkpoint_file.name}")
            print(f"  Dataset: {info['dataset_type']}")
            print(f"  Model: {info['model']}")
            print(f"  Experiment Group: {info['experiment_group']}")
            print(f"  Alpha Version: {info['alpha_version']}")
            print(f"  Finetuning: {info['finetuning']}")
            print(f"  Classes: {get_num_classes(info['experiment_group'])}")
            print()
        except Exception as e:
            print(f"File: {checkpoint_file.name} (Error parsing: {e})")
            print()

def main():
    parser = argparse.ArgumentParser(description="Helper script for running inference with trained models.")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to the model checkpoint")
    parser.add_argument("--input", type=str, required=True,
                       help="Path to input CSV file")
    parser.add_argument("--output", type=str, required=True,
                       help="Path to save predictions")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--max_len", type=int, default=512,
                       help="Maximum sequence length")
    parser.add_argument("--list", action="store_true",
                       help="List all available checkpoints")
    
    args = parser.parse_args()
    
    if args.list:
        list_available_checkpoints()
        return
    
    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint file not found: {args.checkpoint}")
        sys.exit(1)
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    # Run inference
    run_inference(
        checkpoint_path=args.checkpoint,
        input_file=args.input,
        output_file=args.output,
        batch_size=args.batch_size,
        max_len=args.max_len
    )

if __name__ == "__main__":
    main() 