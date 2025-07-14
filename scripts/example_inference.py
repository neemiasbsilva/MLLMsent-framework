#!/usr/bin/env python3
"""
Example script demonstrating how to use the inference functionality.
This script shows different ways to run inference with the trained models.
"""

import os
import pandas as pd
from scripts.inference import inference

def create_sample_data():
    """Create sample data for testing inference."""
    sample_texts = [
        "This is a beautiful sunny day with clear skies.",
        "The movie was terrible and boring.",
        "The food was okay, nothing special.",
        "I absolutely love this new restaurant!",
        "The service was slow and the food was cold.",
        "This product exceeded my expectations.",
        "The weather is quite neutral today.",
        "I'm feeling slightly positive about the outcome.",
        "The results were disappointing.",
        "Everything went perfectly as planned."
    ]
    
    df = pd.DataFrame({"text": sample_texts})
    df.to_csv("sample_data.csv", index=False)
    print("Created sample_data.csv with 10 sample texts")
    return df

def example_bart_inference():
    """Example of running inference with BART model."""
    print("\n" + "="*60)
    print("EXAMPLE: BART Model Inference")
    print("="*60)
    
    # Example checkpoint path (you need to replace with actual path)
    checkpoint_path = "checkpoints/best_checkpoint_gpt4-openai-classify_bart_p5_sigma3_finetuned.pt"
    model_path = "facebook/bart-large-mnli"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please update the checkpoint_path with an actual checkpoint file.")
        return
    
    # Load sample data
    df = pd.read_csv("sample_data.csv")
    texts = df["text"].tolist()
    
    print(f"Running inference on {len(texts)} texts...")
    
    # Run inference
    predictions = inference(
        model_name="bart",
        checkpoint_path=checkpoint_path,
        model_path=model_path,
        texts=texts,
        num_classes=5,  # p5 experiment has 5 classes
        batch_size=8,
        max_len=512
    )
    
    # Display results
    print("\nResults:")
    for i, (text, pred) in enumerate(zip(texts, predictions)):
        sentiment_map = {0: "Negative", 1: "SlightlyNegative", 2: "Neutral", 3: "SlightlyPositive", 4: "Positive"}
        sentiment = sentiment_map.get(pred, f"Unknown({pred})")
        print(f"{i+1:2d}. {text[:50]:<50} -> {sentiment}")

def example_modernbert_inference():
    """Example of running inference with ModernBERT model."""
    print("\n" + "="*60)
    print("EXAMPLE: ModernBERT Model Inference")
    print("="*60)
    
    # Example checkpoint path (you need to replace with actual path)
    checkpoint_path = "checkpoints/best_checkpoint_gpt4-openai-classify_modernbert_p3_sigma5_finetuned.pt"
    model_path = "microsoft/DialoGPT-medium"  # Update with actual ModernBERT path
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please update the checkpoint_path with an actual checkpoint file.")
        return
    
    # Load sample data
    df = pd.read_csv("sample_data.csv")
    texts = df["text"].tolist()
    
    print(f"Running inference on {len(texts)} texts...")
    
    # Run inference
    predictions = inference(
        model_name="modern-bert",
        checkpoint_path=checkpoint_path,
        model_path=model_path,
        texts=texts,
        num_classes=3,  # p3 experiment has 3 classes
        batch_size=8,
        max_len=512
    )
    
    # Display results
    print("\nResults:")
    for i, (text, pred) in enumerate(zip(texts, predictions)):
        sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
        sentiment = sentiment_map.get(pred, f"Unknown({pred})")
        print(f"{i+1:2d}. {text[:50]:<50} -> {sentiment}")

def example_llama_inference():
    """Example of running inference with Llama model."""
    print("\n" + "="*60)
    print("EXAMPLE: Llama Model Inference")
    print("="*60)
    
    # Example checkpoint directory (you need to replace with actual path)
    checkpoint_dir = "checkpoints/llama/gpt4-openai-classify_p2plus_sigma3"
    model_path = "meta-llama/Llama-2-7b-chat-hf"  # Update with actual Llama path
    
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        print("Please update the checkpoint_dir with an actual checkpoint directory.")
        return
    
    # Load sample data
    df = pd.read_csv("sample_data.csv")
    texts = df["text"].tolist()
    
    print(f"Running inference on {len(texts)} texts...")
    
    # Run inference
    predictions = inference(
        model_name="llama",
        checkpoint_path=checkpoint_dir,
        model_path=model_path,
        texts=texts,
        num_classes=2,  # p2plus experiment has 2 classes
        experiment_group="p2plus",
        batch_size=1,  # Llama typically uses batch size of 1
        max_len=512
    )
    
    # Display results
    print("\nResults:")
    for i, (text, pred) in enumerate(zip(texts, predictions)):
        sentiment_map = {0: "Neutral", 1: "Negative"}
        sentiment = sentiment_map.get(pred, f"Unknown({pred})")
        print(f"{i+1:2d}. {text[:50]:<50} -> {sentiment}")

def example_batch_inference():
    """Example of running batch inference on a larger dataset."""
    print("\n" + "="*60)
    print("EXAMPLE: Batch Inference on Large Dataset")
    print("="*60)
    
    # Create a larger sample dataset
    sample_texts = [
        "The customer service was excellent and very helpful.",
        "I'm disappointed with the quality of this product.",
        "The weather is quite pleasant today.",
        "This restaurant has the best food I've ever tasted!",
        "The movie was okay, nothing special.",
        "I'm feeling very frustrated with this situation.",
        "The new features are amazing and work perfectly.",
        "The delivery was late and the package was damaged.",
        "I'm satisfied with the overall experience.",
        "This is the worst purchase I've ever made.",
        "The performance exceeded all expectations.",
        "The interface is confusing and hard to navigate.",
        "I'm neutral about this decision.",
        "The results were outstanding and impressive.",
        "The service was slow and unprofessional."
    ]
    
    df = pd.DataFrame({"text": sample_texts})
    df.to_csv("batch_sample_data.csv", index=False)
    
    # Example checkpoint path (you need to replace with actual path)
    checkpoint_path = "checkpoints/best_checkpoint_gpt4-openai-classify_bart_p5_sigma3_finetuned.pt"
    model_path = "facebook/bart-large-mnli"
    
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Please update the checkpoint_path with an actual checkpoint file.")
        return
    
    texts = df["text"].tolist()
    
    print(f"Running batch inference on {len(texts)} texts...")
    
    # Run inference with larger batch size
    predictions = inference(
        model_name="bart",
        checkpoint_path=checkpoint_path,
        model_path=model_path,
        texts=texts,
        num_classes=5,
        batch_size=16,  # Larger batch size for efficiency
        max_len=512
    )
    
    # Save results
    results_df = df.copy()
    results_df['prediction'] = predictions
    results_df.to_csv("batch_inference_results.csv", index=False)
    
    # Display summary
    print(f"\nResults saved to: batch_inference_results.csv")
    print(f"Prediction distribution: {pd.Series(predictions).value_counts().sort_index().to_dict()}")

def main():
    """Main function to run all examples."""
    print("Inference Examples")
    print("="*60)
    print("This script demonstrates how to use the inference functionality.")
    print("Note: You need to update the checkpoint paths with actual files.")
    print()
    
    # Create sample data
    create_sample_data()
    
    # Run examples (comment out the ones you don't want to run)
    example_bart_inference()
    example_modernbert_inference()
    example_llama_inference()
    example_batch_inference()
    
    print("\n" + "="*60)
    print("All examples completed!")
    print("Check the generated CSV files for detailed results.")

if __name__ == "__main__":
    main() 