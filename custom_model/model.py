import torch
from transformers import BertModel
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomModel(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.encoder = BertModel.from_pretrained('bert-base-uncased')

    def forward(self, inputs, targets):
        layer1 = self.encoder(**inputs)
        cls_token = layer1.last_hidden_state[:, 0, :]
        linear_layer = nn.Linear(768, 1, device=device)
        layer2 = linear_layer(cls_token)
        output = torch.sigmoid(layer2)
        #print(output, output.shape)
        return output

    def forward2(self, input_ids, attention_mask, token_type_ids, targets):
        layer1 = self.encoder(inputs_ids = inputs_ids, \
                attention_mask = attention_mask, token_type_ids = token_type_ids)
        cls_token = layer1.last_hidden_state[:, 0, :]
        linear_layer = nn.Linear(768, 1)
        layer2 = linear_layer(cls_token)
        output = torch.sigmoid(layer2)
        #print(output, output.shape)
        return output

