import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import random
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from scipy import stats
from sklearn.model_selection import KFold
from transformers import AutoImageProcessor, SwinForImageClassification
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from utils.other_utils import load_config
from utils.data_loader import data_loader, load_experiment_data
from scripts.utils_dl import (
    train_one_epoch, validate_one_epoch, compute_metrics,
    save_checkpoint, log_metrics, compute_val_loss_and_preds, compute_val_metrics,
    update_metrics_df, save_metrics_to_csv
)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Use GPU 1 (index 1)
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

class PerceptSentDataset(Dataset):
    def __init__(self, df, image_processor, image_dir, max_len=None):
        self.df = df
        self.image_processor = image_processor
        self.image_dir = image_dir
        self.max_len = max_len
        self.parts = [f"part{i}" for i in range(1, 7)]  # part1 to part6

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['id']
        
        # Try to find image in each part
        for part in self.parts:
            image_path = os.path.join(self.image_dir, part, f"{image_id}.jpg")
            if os.path.exists(image_path):
                try:
                    image = Image.open(image_path).convert('RGB')
                    pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
                    return {
                        'pixel_values': pixel_values,
                        'targets': row['targets']
                    }
                except Exception as e:
                    print(f"Error loading image {image_path}: {e}")
                    continue
        
        # If image not found in any part, return blank image
        print(f"Image {image_id} not found in any part")
        return {
            'pixel_values': torch.zeros((3, 224, 224)),
            'targets': row['targets']
        }

def data_loader(df, image_processor, image_dir, params):
    dataset = PerceptSentDataset(df, image_processor, image_dir)
    return DataLoader(dataset, **params)

def val(log_dir, model, dataloader, loss_fn, kfold, df_metrics, model_name, device):
    """Validation phase."""
    model.eval()
    total_loss, preds, targets = compute_val_loss_and_preds(model, dataloader, loss_fn, device, model_name)
    accuracy, f1, loss = compute_val_metrics(preds, targets, total_loss, dataloader)
    df_metrics = update_metrics_df(df_metrics, kfold, accuracy, f1)
    save_metrics_to_csv(df_metrics, log_dir)
    return df_metrics, preds, targets

def fit(
    model, class_weights, epochs, optimizer,
    train_dl, val_dl,
    log_dir, checkpoint_dir,
    name_arch, fold, model_name,
    device
):
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction="mean")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Log file initialization
    log_file = os.path.join(log_dir, f"training_logs_{fold+1:02d}.txt")
    open(log_file, 'w').close()

    df_metrics = pd.DataFrame([])

    best_f1score = 0
    patience = 10  # Number of epochs to wait for improvement
    patience_counter = 0  # Counter for epochs without improvement

    for epoch in tqdm(range(epochs)):
        # Training phase
        train_loss, preds_train, targets_train = train_one_epoch(model, train_dl, optimizer, loss_fn, device, model_name)
        accuracy_train, f1_train = compute_metrics(preds_train, targets_train)
        train_loss /= len(train_dl)

        # Validation phase
        val_loss, preds_val, targets_val = validate_one_epoch(model, val_dl, loss_fn, device, model_name)
        accuracy_val, f1_val = compute_metrics(preds_val, targets_val)
        val_loss /= len(val_dl)

        # Log metrics
        log_metrics(epoch, epochs, train_loss, accuracy_train, f1_train, val_loss, accuracy_val, f1_val, log_file)

        # Save checkpoint if F1-score improves
        if f1_val > best_f1score:
            best_f1score = save_checkpoint(model, checkpoint_dir, name_arch, f1_val, best_f1score)
            patience_counter = 0  # Reset patience counter
        else:
            patience_counter += 1  # Increment patience counter
        
        # Early stopping check
        if patience_counter >= patience:
            print(f"Validation F1-score has not improved for {patience} epochs. Stopping training.")
            break

        # Update metrics DataFrame
        df_metrics = pd.concat(
            [df_metrics, pd.DataFrame({
                "epoch": [epoch + 1],
                "train_accuracy": [accuracy_train],
                "train_f1_score": [f1_train],
                "val_accuracy": [accuracy_val],
                "val_f1_score": [f1_val],
            })],
            axis=0
        )
        df_metrics.to_csv(os.path.join(log_dir, f"training_logs_{fold+1:02d}.csv"), index=False)

    return model, loss_fn

def train(config, config_path):
    print(f'Train the experiment: {config["experiment_name"]}')

    # Extract hyperparameters
    learning_rate = float(config["learning_rate"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    model_path = config["model_path"]
    log_dir = config["log_dir"]
    checkpoint_dir = config["checkpoint_dir"]
    model_name = config["model_name"]
    image_dir = "/mnt/raid5/neemias/perceptdata"  # Base directory for all parts

    name_arch = model_path.split("/")[-1]
    
    # Load dataset
    alpha_version = int(config_path.split('/')[-2].split('-')[-1][-1])  # 3, 4 or 5
    print(f"Caption LLM: {config_path.split('/')[-2].split('-')[0]}")
    
    if config_path.split('/')[-2].split('-')[0] == "openai":
        dataset_type = "gpt4-openai-classify" 
    elif config_path.split('/')[-2].split('-')[0] == "deepseek":
        dataset_type = "deepseek"
    else:
        dataset_type = "minigpt4-classify"

    experiment_group = config_path.split('/')[-2].split('-')[-2]  # Options: p5, p3, p2plus, p2neg
    print(f"Sigma version: {alpha_version} | experiment_problem: {experiment_group}")
    
    # Load the dataset for the specified experiment
    df = load_experiment_data(alpha_version, dataset_type, experiment_group)

    if df is not None:
        print("Dataset preview:")
        print(df.head())
        # Rename sentiment column to targets for consistency
        df = df.rename(columns={'sentiment': 'targets'})

    train_val_df = df.copy()
    print(f"Number of labels: {len(df.targets.unique())}")
    
    train_params = {"batch_size": batch_size, "shuffle": True}
    val_params = {"batch_size": batch_size, "shuffle": True}

    # Initialize Swin Transformer model and processor
    image_processor = AutoImageProcessor.from_pretrained(model_path)
    

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    df_metrics = pd.DataFrame([])

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_df)):
        print(f"Fold {fold + 1}")
        model = SwinForImageClassification.from_pretrained(
            model_path,
            num_labels=len(df.targets.unique()),
            ignore_mismatched_sizes=True
        )
        train_df = pd.DataFrame(
            {
                "id": train_val_df["id"].iloc[train_idx].to_list(),
                "targets": train_val_df["targets"].iloc[train_idx].to_list(),
            }
        )
        val_df = pd.DataFrame(
            {
                "id": train_val_df["id"].iloc[val_idx].to_list(),
                "targets": train_val_df["targets"].iloc[val_idx].to_list(),
            }
        )

        train_dl = data_loader(train_df, image_processor, image_dir=image_dir, params=train_params)
        val_dl = data_loader(val_df, image_processor, image_dir=image_dir, params=val_params)

        model.to(device)
        optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=1e-6)

        class_size = train_df.targets.value_counts().sort_index().to_list()
        class_weights = torch.Tensor([1 / c for c in class_size]).type(torch.float).to(device)

        model, loss_fn = fit(
            model,
            class_weights,
            epochs,
            optimizer,
            train_dl,
            val_dl,
            log_dir,
            checkpoint_dir,
            name_arch,
            fold,
            model_name,
            device
        )

        df_metrics, y_pred, y_true = val(log_dir, model, val_dl, loss_fn, fold, df_metrics, model_name, device)
        result_df = pd.DataFrame(
            {
                "id": train_val_df["id"].iloc[val_idx].to_list(),
                "target": y_true,
                "prediction": y_pred
            }
        )
        result_df.to_csv(os.path.join(log_dir, f"test_logs_{fold+1:02d}.csv"), index=False)

        del model

    mean_f1 = np.mean(df_metrics["f1_score"].to_numpy())
    confidence_interval = stats.t.interval(
        0.95, len(df_metrics) - 1, loc=mean_f1, scale=stats.sem(df_metrics["f1_score"].to_numpy())
    )

    print(f"Mean F1-score: {mean_f1 * 100:.2f}%")
    print(f"Confidence Interval: {confidence_interval}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Swin Transformer model.")
    parser.add_argument(
        "--config", type=str, default="experiments/experiment1/config.yaml"
    )
    args = parser.parse_args()
    config_path = args.config
    config = load_config(config_path)
    train(config, config_path) 