from dataset_utils import text_dataset
import argparse
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import dataloader

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, default_data_collator

model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    # calculate accuracy using sklearn's function
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
    }

def tokenization(example):
    text = example["text"]
    return tokenizer(text, padding="max_length", truncation=True, max_length=64)

dataset = text_dataset.get_dataset('humor_speech')
dataset = dataset.map(tokenization, remove_columns='text')
dataloader = torch.utils.data.DataLoader(
    dataset,
    collate_fn=default_data_collator,
    batch_size=32,
    shuffle=True,
    pin_memory = True
)

arguments = TrainingArguments(
    output_dir='./results_humor_speech',          # output directory
    num_train_epochs=20,              # total number of training epochs
    per_device_train_batch_size=32,  # batch size per device during training
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=50,
    gradient_accumulation_steps=1,
    evaluation_strategy='steps',
    eval_steps=50,
    save_steps=50,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model='eval_accuracy',
)



trainer = Trainer(
    model = model,                         # the instantiated 🤗 Transformers model to be trained
    args = arguments,                      # training arguments, defined above
    train_dataset = dataset['train'],         # training dataset
    eval_dataset = dataset['valid'],             # evaluation dataset
    compute_metrics = compute_metrics,
)

trainer.train()

model.save_pretrained('./results_humor_speech')