import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

from dataset import CNNDMDataset, DataCollatorForMultiTask
from model import MultitaskBART

mname = 'facebook/bart-base'
data = load_dataset('json', data_files='train.jsonl')
train_dataset = CNNDMDataset(data['train'])
data_collator = DataCollatorForMultiTask(tokenizer=AutoTokenizer.from_pretrained(mname))
train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=data_collator)

model = MultitaskBART(mname)
model.to('cuda')

def train_fn():
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    epochs = 3
    for epoch in range(epochs):
        model.train()
        for (i, batch) in enumerate(train_loader):
            batch = {k: v.to('cuda') for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

if __name__ == '__main__':
    train_fn()