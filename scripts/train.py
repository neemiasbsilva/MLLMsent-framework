import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm
from scipy import stats
from transformers import pipeline, DistilBertTokenizer, AutoTokenizer, TrainingArguments
from transformers import AutoModelForSequenceClassification, AutoConfig
from utils.other_utils import load_config
from utils.data_loader import data_loader
from scripts.utils_dl import (
    load_experiment_data, train_one_epoch, validate_one_epoch, compute_metrics,
    save_checkpoint, log_metrics, compute_val_loss_and_preds, compute_val_metrics,
    update_metrics_df, save_metrics_to_csv
)
from models.model import (
    DistilBERTModel, ModernBERTModel, Llama3
)

from transformers import BartForSequenceClassification, BartTokenizerFast
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.utils import class_weight
from sklearn.model_selection import KFold, train_test_split
from trl import SFTTrainer
from peft import LoraModel, LoraConfig
from datasets import Dataset
from transformers.optimization import AdamW

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use GPU 1 (index 1)

device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}") # Get info about the visible device
else:
    print("Using CPU")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")

def val(log_dir, model, dataloader, loss_fn, kfold, df_metrics, model_name, device):
    """Validation phase."""
    model.eval()
    total_loss, preds, targets = compute_val_loss_and_preds(model, dataloader, loss_fn, device, model_name)
    accuracy, f1, loss = compute_val_metrics(preds, targets, total_loss, dataloader)
    df_metrics = update_metrics_df(df_metrics, kfold, accuracy, f1)
    save_metrics_to_csv(df_metrics, log_dir)
    return df_metrics, preds, targets


def train_llama_qlora(model, tokenizer, train_data, eval_data, log_dir, epochs, batch_size, max_len):
    """Train Llama QLoRA model."""
    peft_config = LoraConfig(
        lora_alpha=8, lora_dropout=0.1, r=32, bias="none", task_type="CAUSAL_LM"
    )

    training_arguments = TrainingArguments(
        output_dir=log_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=1,
        optim="paged_adamw_32bit",
        save_steps=0,
        logging_steps=25,
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="cosine",
        report_to="tensorboard",
        evaluation_strategy="epoch",
        gradient_checkpointing=True,
        eval_accumulation_steps=2,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=eval_data,
        peft_config=peft_config,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_arguments,
        packing=False,
        max_seq_length=max_len,
    )

    trainer.train()
    output_dir = f"{log_dir}/results/trained_model"
    trainer.save_model(output_dir)
    return model, tokenizer


def inference(pipe, prompt):
    """Inference to process a prompt using a pipeline."""
    result = pipe(prompt)
    answer = result[0]['generated_text'].split("=")[-1].strip().strip(".,!?;'\"")
    return answer


def predict(X_test, model, tokenizer):
    """Predict labels for test data."""
    y_pred = []
    pipe = pipeline(
        task="text-generation", model=model, tokenizer=tokenizer, max_new_tokens=1, temperature=0.01, do_sample=True
    )
    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["text"]
        answer = inference(pipe, prompt)
        if answer.isdigit():
            y_pred.append(int(answer))
        else:
            y_pred.append(f"Invalid literal for int(): {answer}")
    return y_pred


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
    patience = 20  # Number of epochs to wait for improvement
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
    max_len = config["max_len"]
    model_path = config["model_path"]
    log_dir = config["log_dir"]
    checkpoint_dir = config["checkpoint_dir"]
    model_name = config["model_name"]

    name_arch = model_path.split("-")[0]
    # Load dataset
    alpha_version = int(config_path.split('/')[-2].split('-')[-1][-1]) # 3, 4 or 5
    # /home/neemias/PerceptSent-LLM-approach/experiments/openai-modernbert-experiment-p2neg-alpha3/config.yaml
    if config_path.split('/')[-2].split('-')[0] == "openai":
        dataset_type = "gpt4-openai-classify" 
    elif config_path.split('/')[-2].split('-')[0] == "deepseek":
        dataset_type = "deepseek"
    else:
        dataset_type = "percept_dataset"

    experiment_group = config_path.split('/')[-2].split('-')[-2]  # Options: p5, p3, p2plus, p2neg
    print(f"Sigma version: {alpha_version} | experiment_problem: {experiment_group}")
    # Load the dataset for the specified experiment
    df = load_experiment_data(alpha_version, dataset_type, experiment_group)

    # Example of further processing if dataset is loaded successfully
    if df is not None:
        print("Dataset preview:")
        print(df.head())

    # train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    train_val_df = df.copy()
    train_params = {"batch_size": batch_size, "shuffle": True}
    val_params = {"batch_size": batch_size, "shuffle": True}

    if model_name in ["distil-bert", "distil-bert-pooling-self-attention"]:
        tokenizer = DistilBertTokenizer.from_pretrained(model_path, do_lower_case=True)
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        df_metrics = pd.DataFrame([])

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_df)):
            print(f"Fold {fold + 1}")

            train_df = pd.DataFrame(
                {
                    "text": train_val_df["text"].iloc[train_idx].to_list(),
                    "sentiment": train_val_df["sentiment"].iloc[train_idx].to_list(),
                }
            )
            val_df = pd.DataFrame(
                {
                    "text": train_val_df["text"].iloc[val_idx].to_list(),
                    "sentiment": train_val_df["sentiment"].iloc[val_idx].to_list(),
                }
            )

            train_dl = data_loader(train_df, tokenizer, max_len, train_params)
            val_dl = data_loader(val_df, tokenizer, max_len, val_params)
            # test_dl = data_loader(test_df, tokenizer, max_len, val_params)

            if model_name == "distil-bert":
                model = DistilBERTModel(model_path, len(df.sentiment.unique()))

            model.to(device)
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=1e-6)

            class_size = train_df.sentiment.value_counts().sort_index().to_list()
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
                    "text": train_val_df["text"].iloc[val_idx].to_list(),
                    "target": y_true,
                    "prediction": y_pred
                }
            )
            result_df.to_csv(os.path.join(log_dir, f"test_logs_{fold+1:02d}.csv"), index=False)

        mean_f1 = np.mean(df_metrics["f1_score"].to_numpy())
        confidence_interval = stats.t.interval(
            0.95, len(df_metrics) - 1, loc=mean_f1, scale=stats.sem(df_metrics["f1_score"].to_numpy())
        )

        print(f"Mean F1-score: {mean_f1 * 100:.2f}%")
        print(f"Confidence Interval: {confidence_interval}")
    
    elif model_name == "modern-bert":
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        df_metrics = pd.DataFrame([])
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_df)):
            print(model_path)
            # Initialize the ModernBERT model
            model = ModernBERTModel(model_path, len(df.sentiment.unique()))
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model.to(device)
            # optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=1e-6)
            optimizer = AdamW(
                model.parameters(),
                lr=learning_rate,
                # betas=(0.9, 0.999),
                # eps=1e-8,  # Controls numerical stability
                weight_decay=1e-6
            )
            # print(model)
            print(f"Fold {fold + 1}")

            train_df = pd.DataFrame(
                {
                    "text": train_val_df["text"].iloc[train_idx].to_list(),
                    "sentiment": train_val_df["sentiment"].iloc[train_idx].to_list(),
                }
            )
            val_df = pd.DataFrame(
                {
                    "text": train_val_df["text"].iloc[val_idx].to_list(),
                    "sentiment": train_val_df["sentiment"].iloc[val_idx].to_list(),
                }
            )

            class_size = train_df.sentiment.value_counts().sort_index().to_list()
            class_weights = torch.Tensor([1 / c for c in class_size]).type(torch.float).to(device)

            train_dl = data_loader(train_df, tokenizer, max_len, train_params)
            val_dl = data_loader(val_df, tokenizer, max_len, val_params)
            # test_dl = data_loader(test_df, tokenizer, max_len, val_params)

            model, loss_fn = fit(
                model, class_weights, epochs, optimizer,
                train_dl, val_dl,
                log_dir, checkpoint_dir,
                name_arch, fold, model_name,
                device
            )

            df_metrics, y_pred, y_true = val(log_dir, model, val_dl, loss_fn, fold, df_metrics, model_name, device)
            result_df = pd.DataFrame(
                {
                    "id": train_val_df["id"].iloc[val_idx].to_list(),
                    "text": train_val_df["text"].iloc[val_idx].to_list(),
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

    elif model_name == "llama-qlora":
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        df_metrics = pd.DataFrame([])

        llama3 = Llama3(model_path)
        model, tokenizer = llama3.get_model()

        prompt = """What is the sentiment of this description? Please choose an answer from \
            {"Positive": 1, "Negative": 0}"""

        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_df)):
            print(f"Fold {fold + 1}")

            train_df = train_val_df.iloc[train_idx]
            val_df = train_val_df.iloc[val_idx]

            train_df["input"] = train_df["text"]
            val_df["input"] = val_df["text"]

            train_df["text"] = train_df.apply(lambda x: f"{prompt}{x['text']}={x['sentiment']}", axis=1)
            val_df["text"] = val_df.apply(lambda x: f"{prompt}{x['text']}=", axis=1)

            train_data = Dataset.from_pandas(train_df)
            eval_data = Dataset.from_pandas(val_df)

            model, tokenizer = train_llama_qlora(
                model, tokenizer, train_data, eval_data, log_dir, epochs, batch_size, max_len
            )

            y_pred_temp = predict(val_df, model, tokenizer)

            y_true_temp = val_df["sentiment"].tolist()
            y_pred = [pred for i, pred in enumerate(y_pred_temp) if isinstance(pred, int)]
            y_true = [y_true_temp[i] for i, pred in enumerate(y_pred_temp) if isinstance(pred, int)]

            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average="weighted")

            df_metrics = pd.concat(
                [
                    df_metrics,
                    pd.DataFrame({"kfold": [fold + 1], "accuracy": [accuracy], "f1_score": [f1]}),
                ],
                axis=0,
            )

            df_metrics.to_csv(os.path.join(log_dir, "test_logs.csv"), index=False)
            result_df = pd.DataFrame(
                {
                    "id": train_val_df["id"].iloc[val_idx].to_list(),
                    "text": train_val_df["text"].iloc[val_idx].to_list(),
                    "target": y_true,
                    "prediction": y_pred
                }
            )
            result_df.to_csv(os.path.join(log_dir, f"test_logs_{fold+1:02d}.csv"), index=False)
            

        mean_f1 = np.mean(df_metrics["f1_score"].to_numpy())
        confidence_interval = stats.t.interval(
            0.95, len(df_metrics) - 1, loc=mean_f1, scale=stats.sem(df_metrics["f1_score"].to_numpy())
        )

        print(f"Mean F1-score: {mean_f1 * 100:.2f}%")
        print(f"Confidence Interval: {confidence_interval}")
    elif model_name == "bart": 
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        df_metrics = pd.DataFrame([])
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_df)):
            # print(model_path)
            print(f"Fold {fold + 1}")

            train_df = pd.DataFrame(
                {
                    "text": train_val_df["text"].iloc[train_idx].to_list(),
                    "sentiment": train_val_df["sentiment"].iloc[train_idx].to_list(),
                }
            )
            val_df = pd.DataFrame(
                {
                    "text": train_val_df["text"].iloc[val_idx].to_list(),
                    "sentiment": train_val_df["sentiment"].iloc[val_idx].to_list(),
                }
            )

            class_size = train_df.sentiment.value_counts().sort_index().to_list()
            class_weights = torch.Tensor([1 / c for c in class_size]).type(torch.float).to(device)
            # Initialize the BART-LARGE MNLI model
            model = BartForSequenceClassification.from_pretrained(
                model_path,
                num_labels=len(class_size),
                ignore_mismatched_sizes=True,
            )
            tokenizer = BartTokenizerFast.from_pretrained(model_path)
            model.to(device)
            # optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=1e-6)
            optimizer = AdamW(
                model.parameters(),
                lr=learning_rate,
                # betas=(0.9, 0.999),
                # eps=1e-8,  # Controls numerical stability
                weight_decay=1e-6
            )
            train_dl = data_loader(train_df, tokenizer, max_len, train_params)
            val_dl = data_loader(val_df, tokenizer, max_len, val_params)
            # test_dl = data_loader(test_df, tokenizer, max_len, val_params)

            model, loss_fn = fit(
                model, class_weights, epochs, optimizer,
                train_dl, val_dl,
                log_dir, checkpoint_dir,
                name_arch, fold, model_name,
                device
            )

            df_metrics, pred, target = val(log_dir, model, val_dl, loss_fn, fold, df_metrics, model_name, device)
            result_df = pd.DataFrame(
                {
                    "id": train_val_df["id"].iloc[val_idx].to_list(),
                    "text": train_val_df["text"].iloc[val_idx].to_list(),
                    "target": target,
                    "prediction": pred
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
    else:
        raise ValueError(f"Unsupported model name: {model_name}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the DL model.")
    parser.add_argument(
        "--config", type=str, default="experiments/experiment1/config.yaml"
    )
    args = parser.parse_args()
    config_path = args.config
    config = load_config(config_path)
    train(config, config_path)
