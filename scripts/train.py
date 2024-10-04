import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from utils.other_utils import load_config
from utils.data_loader import data_loader
from transformers import DistilBertTokenizer, BertTokenizer, AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
from models.model import DistilBERTModel, BERTFinetuningModel, RoBERTaFinetuningModel
from transformers import BartForSequenceClassification, BartTokenizerFast
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim.lr_scheduler import ExponentialLR
from sklearn.utils import class_weight
from sklearn.model_selection import KFold, train_test_split
from transformers import AdamW
from evaluate import evaluate
import torch.nn.functional as F

# from test import 

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_dataframe(path):
    return pd.read_csv(path)


# def loss_fn(class_weights, outputs, targets):
#     return torch.nn.CrossEntropyLoss(weight=class_weights,reduction='mean')(outputs, targets)

def val(log_dir, model, dataloader, loss_fn, kfold, df_metrics, model_name):
    ## Validation phase
    model.eval()
    total_loss = 0.0
    preds = []
    targets = []
    with torch.no_grad():
        for _, data in enumerate(dataloader, 0):
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            targets_ = data["targets"].to(device)
            token_type_id = data["token_type_id"].to(device)

            outputs = model(ids, mask, token_type_id)
            if model_name == "bart":
                outputs = outputs.logits
            loss = loss_fn(outputs, targets_)
            total_loss += loss.item()
            preds.extend(torch.argmax(outputs, axis=1).tolist())
            targets.extend(targets_.tolist())

    accuracy = accuracy_score(targets, preds)
    f1 = f1_score(targets, preds, average="weighted")
    loss = total_loss/len(dataloader)
    
    df_metrics = pd.concat([df_metrics, pd.DataFrame({"kfold": [kfold+1], "accuracy": [accuracy], "f1_score": [f1]
                                                         })], axis=0)
    df_metrics.to_csv(os.path.join(log_dir, f"test_logs.csv"), index=False)

    return df_metrics

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Distância euclidiana entre as duas representações
        euclidean_distance = F.pairwise_distance(output1, output2)
        
        # Perda contrativa
        loss = torch.mean(
            (1 - label) * torch.pow(euclidean_distance, 2) +
            label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2)
        )
        return loss


def fit(model, class_weights, epochs, optimizer, train_dl, val_dl, log_dir, checkpoint_dir, name_arch, fold, model_name):
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='mean')
    # for param in model.distilbert.parameters():
    # for param in model.distilbert.parameters():
    # for param in model.distilbert.parameters():
        # param.requires_grad = False
    model.to(device)
    torch.manual_seed(42)
    np.random.seed(42)
    log_file = os.path.join(log_dir, f"training_logs_{fold+1:02d}.txt")
    df_metrics = pd.DataFrame([])
    best_acc = 0
    stop_train = 0
    for epoch in tqdm(range(epochs)):
        ## Training phase
        model.train()
        total_loss_train = 0.0
        preds_train = []
        targets_train = []
        for _, data in enumerate(train_dl, 0):
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            targets = data["targets"].to(device)
            token_type_id = data["token_type_id"].to(device)
            # Zero Gradients for every batch!
            optimizer.zero_grad()
            outputs = model(ids, mask, token_type_id)
            # Compute the loss and its gradients
            if model_name == "bart":
                outputs = outputs.logits
            # print(outputs.logits)
            # exit()
            loss = loss_fn(outputs, targets)
            loss.backward()
            
            # Adjust learning weights
            optimizer.step()
            # scheduler.step()
            total_loss_train += loss.item()
            preds_train.extend(torch.argmax(outputs, axis=1).tolist())
            targets_train.extend(targets.tolist())

        ## Validation phase
        model.eval()
        total_loss_val = 0.0
        preds_val = []
        targets_val = []
        with torch.no_grad():
            for _, data in enumerate(val_dl, 0):
                ids = data["ids"].to(device)
                mask = data["mask"].to(device)
                targets = data["targets"].to(device)
                token_type_id = data["token_type_id"].to(device)

                outputs = model(ids, mask, token_type_id)
                if model_name == "bart":
                    outputs = outputs.logits
                loss_val = loss_fn(outputs, targets)
                total_loss_val += loss_val.item()
                preds_val.extend(torch.argmax(outputs, axis=1).tolist())
                targets_val.extend(targets.tolist())

        
        ## Calculate metrics
        accuracy_train = accuracy_score(targets_train, preds_train)
        f1_train = f1_score(targets_train, preds_train, average="weighted")
        loss_train = total_loss_train/len(train_dl)

        accuracy_val = accuracy_score(targets_val, preds_val)
        f1_val = f1_score(targets_val, preds_val, average="weighted")
        loss_val = total_loss_val/len(val_dl)

        # Write logs to file
        # print(f"Epoch {epoch+1}/{epochs}:")
        # print(f"Train Loss: {loss_train:.4f}, Accuracy: {accuracy_train:.4f}, F1-score: {f1_train:.4f}")
        # print(f"Val Loss: {loss_val:.4f}, Accuracy: {accuracy_val:.4f}, F1-score: {f1_val:.4f}")
        log_entry = (
            f"Epoch {epoch+1}/{epochs}: \n"
            f"Train Loss: {loss_train:.4f}, Accuracy: {accuracy_train:.4f}, "
            f"F1-score: {f1_train:.4f}\n"
            f"Validation Loss: {loss_val:.4f}, Accuracy: {accuracy_val:.4f}, "
            f"F1-score: {f1_val:.4f}\n"
        )
        with open(log_file, 'a') as f:
            f.write(log_entry)

        
        if ( stop_train == 10): 
            print("Validation accuracy don't change for 10 epochs, finish training")
            break
        df_metrics = pd.concat([df_metrics, pd.DataFrame({"epoch": [epoch+1], "train_accuracy": [accuracy_train], "train_f1_score": [f1_train],
                                                         "val_accuracy": [accuracy_val], "val_f1_score": [f1_val]
                                                         })], axis=0)
        df_metrics.to_csv(os.path.join(log_dir, f"training_logs_{fold+1:02d}.csv"), index=False)

        # scheduler.step()
        # Save checkpoint
        if (best_acc < accuracy_val):
            best_acc = accuracy_val
            stop_train = 0
            checkpoint_path = os.path.join(checkpoint_dir, f"best_checkpoint_{name_arch.split('/')[0]}.pt")
            torch.save(model.state_dict(), checkpoint_path)
        stop_train += 1

    return model, loss_fn


def train(config):
    print(f'Train the experiment: {config["experiment_name"]}')

    # get hyperparameters
    learning_rate = float(config["learning_rate"])
    batch_size = config["batch_size"]
    epochs = config["epochs"]
    max_len = config["max_len"]
    model_path = config["model_path"]
    log_dir = config["log_dir"]
    checkpoint_dir = config['checkpoint_dir']
    model_name = config['model_name']

    step_size = 1
    gamma = 0.1
    name_arch = model_path.split('-')[0]


    # load_datasets and preprocessing pytorch dataloader
    ### Load experiments for alpha 3
    # df = load_dataframe("data/percept_dataset_alpha3_p5.csv")
    # df = load_dataframe("data/gpt4-openai-classify/percept_dataset_alpha3_p5.csv")
    # df = load_dataframe("data/percept_dataset_alpha3_p3.csv")
    # df = load_dataframe("data/gpt4-openai-classify/percept_dataset_alpha3_p3.csv")
    # df = load_dataframe("data/percept_dataset_alpha3_p2plus.csv")
    # df = load_dataframe("data/gpt4-openai-classify/percept_dataset_alpha3_p2plus.csv")
    # df = load_dataframe("data/percept_dataset_alpha3_p2neg.csv")
    # df = load_dataframe("data/gpt4-openai-classify/percept_dataset_alpha3_p2neg.csv")

    ### Load experiments for alpha 4
    # df = load_dataframe("data/percept_dataset_alpha4_p5.csv")
    # df = load_dataframe("data/gpt4-openai-classify/percept_dataset_alpha4_p5.csv")
    # df = load_dataframe("data/percept_dataset_alpha4_p3.csv")
    # df = load_dataframe("data/gpt4-openai-classify/percept_dataset_alpha4_p3.csv")
    # df = load_dataframe("data/percept_dataset_alpha4_p2plus.csv")
    # df = load_dataframe("data/gpt4-openai-classify/percept_dataset_alpha4_p2plus.csv")
    # df = load_dataframe("data/percept_dataset_alpha4_p2neg.csv")
    df = load_dataframe("data/gpt4-openai-classify/percept_dataset_alpha4_p2neg.csv")

    ### Load experiments for alpha 5
    # df = load_dataframe("data/percept_dataset_alpha5_p5.csv")
    # df = load_dataframe("data/percept_dataset_alpha5_p3.csv")
    # df = load_dataframe("data/percept_dataset_alpha5_p2plus.csv")
    # df = load_dataframe("data/percept_dataset_alpha5_p2neg.csv")

    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    test_df = pd.DataFrame({"text": test_df.text.to_list(), "sentiment": test_df.sentiment.to_list()})
    
    # train_df = load_dataframe("data/train/aug_train.csv")

    train_params = {
        "batch_size": batch_size,
        "shuffle": True,
    }
    # val_df = load_dataframe("data/validation/val.csv")
    
    val_params = {
        "batch_size": batch_size,
        "shuffle": True
    }

    if (model_name == "distil-bert"):
        # # Define the tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(
            model_path,
            do_lower_case=True
        )
        # tokenizer = BertTokenizer.from_pretrained(
        #     model_path,
        #     # do_lower_case=True
        # )
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # config = AutoConfig.from_pretrained(model_path)
        
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        df_metrics = pd.DataFrame([])
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_df)):
            print(f"Fold {fold + 1}")
            train_df = pd.DataFrame({"text": train_val_df["text"].iloc[train_idx].to_list(), 
                                     "sentiment": train_val_df["sentiment"].iloc[train_idx].to_list()})
            val_df = pd.DataFrame({"text": train_val_df["text"].iloc[val_idx].to_list(), 
                                     "sentiment": train_val_df["sentiment"].iloc[val_idx].to_list()})
            
            train_dl = data_loader(train_df, tokenizer, max_len, train_params)
            val_dl = data_loader(val_df, tokenizer, max_len, val_params)
            test_dl = data_loader(test_df, tokenizer, max_len, val_params)

            model = DistilBERTModel(model_path, len(df.sentiment.value_counts()))
            # model = BERTFinetuningModel(model_path)
            # model = RoBERTaFinetuningModel(model_path)
            model.to(device)
            print(model)

            # optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate, weight_decay=0.01)
            optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=1e-6)
            # optimizer = torch.optim.SGD(params=model.parameters(), lr=learning_rate, momentum=.9)
            # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, steps_per_epoch=len(train_dl), epochs=epochs)
            
            class_size = train_df.sentiment.value_counts().sort_index().to_list()
            if (len(class_size) == 3):
                class_weights = torch.Tensor([1/class_size[0], 1/class_size[1], 1/class_size[2]]).type(torch.float).to(device)
            elif (len(class_size) == 2):
                class_weights = torch.Tensor([1/class_size[0], 1/class_size[1]]).type(torch.float).to(device)
            else:
                class_weights = torch.Tensor([1/class_size[0], 1/class_size[1], 1/class_size[2], 1/class_size[3], 1/class_size[4]]).type(torch.float).to(device)

                                        

            model, loss_fn = fit(model, class_weights, epochs, optimizer, train_dl, val_dl, log_dir, checkpoint_dir, name_arch, fold, model_name)

            df_metrics = val(log_dir, model, test_dl, loss_fn, fold, df_metrics, model_name)
        
        print(f'Mean F1-score: {np.mean(df_metrics["f1_score"].to_numpy())*100:.2f}%')
    else:
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer = BartTokenizerFast.from_pretrained(model_path)
        # config_model = AutoConfig.from_pretrained(model_path)

        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        df_metrics = pd.DataFrame([])
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_df)):
            print(f"Fold {fold + 1}")
            train_df = pd.DataFrame({"text": train_val_df["text"].iloc[train_idx].to_list(), 
                                     "sentiment": train_val_df["sentiment"].iloc[train_idx].to_list()})
            val_df = pd.DataFrame({"text": train_val_df["text"].iloc[val_idx].to_list(), 
                                     "sentiment": train_val_df["sentiment"].iloc[val_idx].to_list()})
            
            train_dl = data_loader(train_df, tokenizer, max_len, train_params)
            val_dl = data_loader(val_df, tokenizer, max_len, val_params)
            test_dl = data_loader(test_df, tokenizer, max_len, val_params)

            class_size = train_df.sentiment.value_counts().sort_index().to_list()


            if (len(class_size) == 3):
                class_weights = torch.Tensor([1/class_size[0], 1/class_size[1], 1/class_size[2]]).type(torch.float).to(device)
            elif (len(class_size) == 2):
                class_weights = torch.Tensor([1/class_size[0], 1/class_size[1]]).type(torch.float).to(device)
            else:
                class_weights = torch.Tensor([1/class_size[0], 1/class_size[1], 1/class_size[2], 1/class_size[3], 1/class_size[4]]).type(torch.float).to(device)
        
            model = BartForSequenceClassification.from_pretrained('facebook/bart-large-mnli', num_labels=len(class_size), ignore_mismatched_sizes=True)
            model.to(device)
            print(model)  

            optimizer = torch.optim.AdamW(params=model.parameters(), lr=learning_rate, weight_decay=1e-6)            

            model, loss_fn = fit(model, class_weights, epochs, optimizer, train_dl, val_dl, log_dir, checkpoint_dir, name_arch, fold, model_name)

            df_metrics = val(log_dir, model, test_dl, loss_fn, fold, df_metrics, model_name)
        
        print(f'Mean F1-score: {np.mean(df_metrics["f1_score"].to_numpy())*100:.2f}%')

        # data = train_df.text
        # target = train_df.sentiment
        # set_name = "Train"
        # evaluate(tokenizer, model, config_model, data, target, log_dir, set_name)
        # data = val_df.text
        # target = val_df.sentiment
        # set_name = "Validation"
        # evaluate(tokenizer, model, config_model, data, target, log_dir, set_name)


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train the DL model.")
    parser.add_argument("--config", type=str, default="experiments/experiment1/config.yaml")
    args = parser.parse_args()

    # Load configuration
    config_path = args.config
    config = load_config(config_path)

    train(config)