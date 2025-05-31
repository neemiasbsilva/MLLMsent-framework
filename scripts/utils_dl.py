import os
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score


def load_dataframe(file_path):
    """
    Simulate loading a dataset from a CSV file.

    Parameters:
        file_path (str): Path to the dataset file.

    Returns:
        DataFrame: A pandas DataFrame with the loaded data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Successfully loaded dataset from: {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None


def load_experiment_data(alpha_version, dataset_type, experiment_group):
    """
    Load the dataset dynamically based on experiment parameters.

    Parameters:
        alpha_version (int): The alpha version (e.g., 3, 4, 5).
        dataset_type (str): Dataset type, e.g., "percept_dataset" or "gpt4-openai-classify".
        experiment_group (str): Experiment group, e.g., "p5", "p3", "p2plus", "p2neg".

    Returns:
        DataFrame: The loaded dataset.
    """
    # Build the dynamic file path
    file_path = f"data/{dataset_type}/percept_dataset_alpha{alpha_version}_{experiment_group}.csv"

    # Load the dataset
    df = load_dataframe(file_path)
    print(f"Loaded dataset from: {file_path}")
    return df


def compute_loss(outputs, targets, loss_fn, model_name):
    return loss_fn(outputs, targets)

def train_one_epoch(model, train_dl, optimizer, loss_fn, device, model_name):
    model.train()
    total_loss = 0.0
    preds, targets = [], []
    
    for _, data in enumerate(train_dl):
        ids = data["ids"].to(device)
        mask = data["mask"].to(device)
        targets_batch = data["targets"].to(device)
        token_type_id = data["token_type_id"].to(device)
        
        optimizer.zero_grad()
        outputs = model(ids, mask, token_type_id)

        if model_name == "bart":
            outputs = outputs.logits
        loss = compute_loss(outputs, targets_batch, loss_fn, model_name)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        
        preds.extend(torch.argmax(outputs, axis=1).tolist())
        targets.extend(targets_batch.tolist())
    
    return total_loss, preds, targets

def validate_one_epoch(model, val_dl, loss_fn, device, model_name):
    model.eval()
    total_loss = 0.0
    preds, targets = [], []

    with torch.no_grad():
        for _, data in enumerate(val_dl):
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            targets_batch = data["targets"].to(device)
            token_type_id = data["token_type_id"].to(device)

            outputs = model(ids, mask, token_type_id)

            if model_name == "bart":
                outputs = outputs.logits
            loss = compute_loss(outputs, targets_batch, loss_fn, model_name)

            total_loss += loss.item()
            preds.extend(torch.argmax(outputs, axis=1).tolist())
            targets.extend(targets_batch.tolist())

    return total_loss, preds, targets


def compute_metrics(preds, targets):
    accuracy = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="weighted")
    return accuracy, f1

def save_checkpoint(model, checkpoint_dir, name_arch, f1_val, best_f1score):
    if f1_val > best_f1score:
        best_f1score = f1_val
        checkpoint_path = os.path.join(
            checkpoint_dir, f"best_checkpoint_{name_arch.split('/')[0]}.pt"
        )
        torch.save(model.state_dict(), checkpoint_path)
    return best_f1score

def log_metrics(epoch, epochs, train_loss, train_accuracy, train_f1, val_loss, val_accuracy, val_f1, log_file):
    log_entry = (
        f"Epoch {epoch+1}/{epochs}: \n"
        f"Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
        f"F1-score: {train_f1:.4f}\n"
        f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, "
        f"F1-score: {val_f1:.4f}\n"
    )
    with open(log_file, "a") as f:
        f.write(log_entry)

def compute_val_loss_and_preds(model, dataloader, loss_fn, device, model_name):
    total_loss = 0.0
    preds, targets = [], []
    
    with torch.no_grad():
        for _, data in enumerate(dataloader):
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            targets_batch = data["targets"].to(device)
            token_type_id = data["token_type_id"].to(device)

            outputs = model(ids, mask, token_type_id)
            if model_name == "bart":
                outputs = outputs.logits

            loss = loss_fn(outputs, targets_batch)
            total_loss += loss.item()
            preds.extend(torch.argmax(outputs, axis=1).tolist())
            targets.extend(targets_batch.tolist())
    
    return total_loss, preds, targets

def compute_val_metrics(preds, targets, total_loss, dataloader):
    accuracy = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="weighted")
    loss = total_loss / len(dataloader)
    return accuracy, f1, loss

def update_metrics_df(df_metrics, kfold, accuracy, f1):
    new_metrics = pd.DataFrame({
        "kfold": [kfold + 1],
        "accuracy": [accuracy],
        "f1_score": [f1]
    })
    df_metrics = pd.concat([df_metrics, new_metrics], axis=0)
    return df_metrics

def save_metrics_to_csv(df_metrics, log_dir):
    df_metrics.to_csv(os.path.join(log_dir, f"test_logs.csv"), index=False)
