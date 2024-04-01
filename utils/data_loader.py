import torch
from torch.utils.data import Dataset, DataLoader

import yaml

class SentimentDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    
    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, index):
        text = str(self.data.text[index])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids = True,
            return_tensors='pt'
        )
        ids = inputs["input_ids"].flatten()
        mask = inputs["attention_mask"].flatten()
        token_type_id = inputs["token_type_ids"].flatten()
        return {
            "ids": ids,
            "mask": mask,
            "token_type_id": token_type_id,
            "targets": torch.tensor(self.data.sentiment[index], dtype=torch.long)
        }
    

def data_loader(dataframe, tokenizer, max_len, params):
    ds = SentimentDataset(dataframe, tokenizer, max_len)
    dl = DataLoader(ds, **params)
    return dl