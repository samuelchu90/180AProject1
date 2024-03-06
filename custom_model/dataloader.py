from torch.utils.data import DataLoader, RandomSampler
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_dataloader(dataset):
    sampler = RandomSampler(dataset)

    batch_size = 16
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader


def prepare_inputs(seqs, tokenizer):
    tokenized_seq = tokenizer(seqs, padding=True, truncation=True, return_tensors='pt').to(device)
    tokenized_seq = {'input_ids': tokenized_seq['input_ids'].to(device), 'token_type_ids': tokenized_seq['token_type_ids'].to(device), \
            'attention_mask': tokenized_seq['attention_mask'].to(device)}
    return tokenized_seq
