from model import CustomModel
from transformers import BertTokenizer
from dataloader import get_dataloader, prepare_inputs
from tqdm import tqdm as progress_bar
import torch
from dataset import getData, AminoAcidDataset
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
            tokenized_seqs = prepare_inputs(seqs, tokenizer) 
            labels = labels.to(device)
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
        tokenized_seq = prepare_inputs(seqs, tokenizer)
        labels = labels.to(device)
        print(tokenized_seq)
        print(f'labels_device{labels.device}')
        print(next(model.parameters()).device)
        probabilities = model(tokenized_seq, labels)
        predictions = torch.round(probabilities)
        #print(predictions, labels)
        num_correct += ((predictions == labels).sum()).item()
        total += len(predictions)
        #print(num_correct)
        #print(total)
    print(f'Accuracy is {num_correct}/{total}, {num_correct/total}')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = CustomModel(tokenizer).to(device)
run_eval(model, dataset)
train(model, dataset, tokenizer)
