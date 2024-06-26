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
    n_epochs = 15 
    #criterion = nn.CrossEntropyLoss() #old
    criterion = nn.BCELoss() #new

    train_dataloader = get_dataloader(dataset)

    optimizer = optim.Adam(model.parameters(), lr=0.0000001)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    for epoch in range(n_epochs):
        losses = 0
        model.train()

        for seqs, labels in progress_bar(train_dataloader):
            tokenized_seqs = prepare_inputs(seqs, tokenizer) 
            labels = labels.to(device)
            #predictions and labels need to have the same dtype: float32
            labels = labels.float()
            predictions = model(tokenized_seqs, labels)
            loss = criterion(predictions, labels)
            losses += loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        print(f'Epoch {epoch} - Loss: {losses}')
        #run_eval(model, dataset)



def run_eval(model, dataset):
    print("<<Eval>>")
    model.eval()
    dataloader = get_dataloader(dataset)

    num_correct = 0
    total = 0
    for seqs, labels in progress_bar(dataloader):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenized_seq = prepare_inputs(seqs, tokenizer)
        labels = labels.to(device)
        probabilities = model(tokenized_seq, labels) 
        predictions = torch.round(probabilities)
        num_correct += (predictions == labels).sum()
        total += len(predictions)
    print(f'Accuracy is {num_correct}/{total}, {num_correct/total}')


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = CustomModel(tokenizer).to(device)
#run_eval(model, dataset)
train(model, dataset, tokenizer)
