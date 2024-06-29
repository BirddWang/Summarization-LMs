import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

#torch.backends.cuda.matmul.allow_tf32 = True
#torch.backends.cuda.allow_tf32 = True

from transformers import BartForConditionalGeneration, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class DataCollatorForMultiTask:
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        
        explain_features = [{
            'input': f['explain_inputs'],
            'target': f['explain_targets']
        } for f in features]

        summary_features = [{
            'input': f['summary_inputs'],
            'target': f['summary_targets'] 
        } for f in features]

        explain_inputs = self.tokenizer([f['input'] for f in explain_features], padding=True, truncation=True, return_tensors="pt", max_length=1024)
        explain_targets = self.tokenizer([f['target'] for f in explain_features], padding=True, truncation=True, return_tensors="pt", max_length=1024)

        summary_inputs = self.tokenizer([f['input'] for f in summary_features], padding=True, truncation=True, return_tensors="pt", max_length=1024)
        summary_targets = self.tokenizer([f['target'] for f in summary_features], padding=True, truncation=True, return_tensors="pt", max_length=1024)
        
        explain_labels = explain_targets['input_ids'].clone()
        explain_labels[explain_labels == self.tokenizer.pad_token_id] = 0

        summary_labels = summary_targets['input_ids'].clone()
        summary_labels[summary_labels == self.tokenizer.pad_token_id] = 0

        return {
            'explain_input_ids': explain_inputs['input_ids'], 
            'explain_attention_mask': explain_inputs['attention_mask'],
            'explain_labels': explain_labels,
            'summary_input_ids': summary_inputs['input_ids'],
            'summary_attention_mask': summary_inputs['attention_mask'],
            'summary_labels': summary_labels,
        }

class CNNDMDataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data_len = len(data['article'])

        self.explain_inputs = ["Explain: " + article for article in data['article']]
        self.summary_inputs = ["Summary: " + article for article in data['article']]
        self.explain_targets = data['rationale']
        self.summary_targets = data['summary']
    def __len__(self):
        return self.data_len

    def __getitem__(self, index) -> Any:
        return {
            'explain_inputs': self.explain_inputs[index],
            'summary_inputs': self.summary_inputs[index],
            'explain_targets': self.explain_targets[index],
            'summary_targets': self.summary_targets[index]
        }

class MultitaskBART(nn.Module):
    def __init__(self, mname) -> None:
        super().__init__()
        self.bart = BartForConditionalGeneration.from_pretrained(mname)

    def forward(self, explain_input_ids=None, explain_attention_mask=None, explain_labels=None,
                summary_input_ids=None, summary_attention_mask=None, summary_labels=None):
        explain_loss = 0
        summary_loss = 0

        if explain_labels is not None:
            explain_outputs = self.bart(input_ids=explain_input_ids, 
                                        attention_mask=explain_attention_mask,
                                        labels=explain_labels)
            explain_loss = explain_outputs.loss
        if summary_labels is not None:
            summary_outputs = self.bart(input_ids=summary_input_ids,
                                        attention_mask=summary_attention_mask,
                                        labels=summary_labels)
            summary_loss = summary_outputs.loss
        if explain_loss is not None and summary_loss is not None:
            total_loss = explain_loss + summary_loss
            return {"loss": total_loss, "explain_loss": explain_loss, "summary_loss": summary_loss}
        elif explain_loss is not None:
            return {"loss": explain_loss, "explain_loss": explain_loss}
        elif summary_loss is not None:
            return {"loss": summary_loss, "summary_loss": summary_loss}
        else:
            raise ValueError("No loss calculated")
        
class MultitaskTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        print(loss)
        return (loss, outputs) if return_outputs else loss

mname = 'facebook/bart-base'
data = load_dataset('json', data_files='temp.jsonl')
train_dataset = CNNDMDataset(data['train'])
data_collator = DataCollatorForMultiTask(tokenizer=AutoTokenizer.from_pretrained(mname))
train_loader = DataLoader(train_dataset, batch_size=4, collate_fn=data_collator)

model = MultitaskBART(mname)

def train_fn():
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    epochs = 3
    for epoch in range(epochs):
        model.train()
        for (i, batch) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs["loss"]
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")

train_fn()
'''
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    save_steps=1000,
    logging_steps=1000,
    learning_rate=5e-5,
    save_total_limit=2,
)


trainer = MultitaskTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=AutoTokenizer.from_pretrained(mname),
    data_collator=data_collator,
)
trainer.train()
'''
