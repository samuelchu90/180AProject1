from model import CustomModel
from transformers import BertTokenizer
from dataloader import get_dataloader
from tqdm import tqdm as progress_bar
import torch
from dataset import getData, AminoAcidDataset
from torch import nn, optim


sequences, labels = getData()
dataset = AminoAcidDataset(sequences, labels)


def train(model, dataset, tokenizer):
    print("<<Training>>")
    n_epochs = 3
    criterion = nn.CrossEntropyLoss()
    train_dataloader = get_dataloader(dataset)

    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(n_epochs):
        losses = 0
        model.train()

        for seqs, labels in progress_bar(train_dataloader):
            inputs = tokenizer(seqs) 
            predictions = model(inputs, labels)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            #print(tokenizer(seqs), len(tokenizer(seqs)))



def run_eval(model, dataset):
    print("<<Eval>>")
    dataloader = get_dataloader(dataset)

    num_correct = 0
    total = 0
    for seqs, labels in progress_bar(dataloader):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized_seq = tokenizer(seqs, padding=True, truncation=True, return_tensors='pt')
        #print(seqs[0], len(seqs[0]))
        #print(tokenized_seq['input_ids'][0], len(tokenized_seq['input_ids'][0]))
        #might be a problem that sequence length != tokenized_seq
        probabilities = model(tokenized_seq, labels)
        predictions = torch.round(probabilities)
        #print(predictions, labels)
        num_correct += ((predictions == labels).sum()).item()
        total += len(predictions)
        #print(num_correct)
        #print(total)
    print(f'Accuracy is {num_correct}/{total}, {num_correct/total}')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = CustomModel(tokenizer)
run_eval(model, dataset)
train(model, dataset, tokenizer)