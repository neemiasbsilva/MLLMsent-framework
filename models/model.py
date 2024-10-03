import torch
from transformers import DistilBertModel
from transformers import BertModel
from transformers import AutoModelForSequenceClassification

class DistilBERTModel(torch.nn.Module):
    def __init__(self, bert_path, class_size):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained(bert_path)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(768, 1024),
            # torch.nn.BatchNorm1d(1024),
            # torch.nn.ReLU(),
            # torch.nn.Dropout(0.5),
            # torch.nn.Linear(1024, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, class_size)
        )
    

    def forward(self, ids, mask, token_type_id):
        output = self.distilbert(ids, attention_mask=mask)
        # shape: (batch_size, seq_length, bert_hidden_dim)
        last_hidden_state = output.last_hidden_state 
        
        # indexing the tensor containing the hidden representations
        CLS_token_state = last_hidden_state[:, 0, :]

        # passing this representation through our custom classifier
        out = self.classifier(CLS_token_state)

        return out
    

class BERTFinetuningModel(torch.nn.Module):
    def __init__(self, bert_path):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_path)
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(1024, 768),
            torch.nn.ReLU(),
            torch.nn.Linear(768, 3)
        )
        
    

    def forward(self, ids, mask, token_type_id):
        output = self.bert(ids, mask, token_type_id)
        # shape: (batch_size, seq_length, bert_hidden_dim)
        last_hidden_state = output.last_hidden_state

        # indexing the tensor containing the hidden representations
        CLS_token_state = last_hidden_state[:, 0, :]
        
        # passing this representation through our custom classifier
        out = self.classifier(CLS_token_state)

        return out

class RoBERTaFinetuningModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.roberta = AutoModelForSequenceClassification.from_pretrained(model_path)
        # self.pre_classifier = torch.nn.Linear(768, 768)
        # self.dropout = torch.nn.Dropout(0.1)
        # self.relu = torch.nn.ReLU()
        # self.out = torch.nn.Linear(768, 3)
    
    def forward(self, ids, mask, token_type_id):
        return self.roberta(ids, mask).logits
        # print(output)
        # hidden_state = output[0]
        # pooled_output = output[:, 0]
        # x = self.pre_classifier(pooled_output)
        # x = self.dropout(self.relu(x))
        # return out


