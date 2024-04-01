import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
import os
from tqdm import tqdm

def evaluate(tokenizer, model, config, data, target, log_dir, set_name):

    log_file = os.path.join(log_dir, "zero_shot_learning_metrics.txt")
    pred = []
    for text in tqdm(data, desc=f"{set_name} progress"):
        encoded_input = tokenizer(text, max_length=512, truncation=True, return_tensors='pt')
        output = model(**encoded_input)
        scores = torch.argmax(output.logits, axis=1).tolist()
        pred.append(scores)


    accuracy = accuracy_score(target, pred)
    f1 = f1_score(target, pred, average="weighted")
    log_entry = (
            f"{set_name} "
            f"Accuracy: {accuracy:.4f}, "
            f"F1-score: {f1:.4f}\n"
    )
    with open(log_file, 'a') as f:
        f.write(log_entry)
