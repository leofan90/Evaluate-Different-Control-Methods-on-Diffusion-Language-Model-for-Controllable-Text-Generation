from dataset_utils import text_dataset
import argparse
from sklearn.metrics import accuracy_score
import torch
import torch.utils.data
import numpy as np

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, default_data_collator

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

model.load_state_dict(torch.load('./results_humor_speech/pytorch_model.bin'))
model.cuda()

sample_dir = "saved_models/humor_speech/epochs_10000_eval_test"
with open("{}/eval42-cond0_nucleus-sample-1.txt".format(sample_dir), 'r', encoding='utf-8') as f:
    samples_cond0 = [line.strip() for line in f.readlines()]
with open("{}/eval42-cond1_nucleus-sample-1.txt".format(sample_dir), 'r', encoding='utf-8') as f:
    samples_cond1 = [line.strip() for line in f.readlines()]

texts = samples_cond0 + samples_cond1
labels = [0] * len(samples_cond0) + [1] * len(samples_cond1)

text_encodings = tokenizer(texts, padding="max_length", truncation=True, max_length=64)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
    
    def __len__(self):
        return len(self.labels)
    
dataset = TextDataset(text_encodings, labels)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
    shuffle=False,
    pin_memory = True
)

model.eval()

with torch.no_grad():
    preds = []
    for batch in dataloader:
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs[0]
        preds.append(logits.argmax(-1).cpu().numpy())
    preds = np.concatenate(preds, axis=0)

print(accuracy_score(labels, preds))