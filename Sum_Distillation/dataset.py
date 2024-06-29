from torch.utils.data import Dataset
from typing import List, Dict, Any

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