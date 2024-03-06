import torch
from transformers import BertModel
from torch import nn

class CustomModel(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, inputs, targets):
        layer1 = self.encoder(**inputs)
        cls_token = layer1.last_hidden_state[:, 0, :]
        linear_layer = nn.Linear(768, 1)
        layer2 = linear_layer(cls_token)
        output = torch.sigmoid(layer2)
        #print(output, output.shape)
        return output 
