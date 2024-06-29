import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration

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