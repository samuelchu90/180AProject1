from torch.utils.data import Dataset

def getData():
    amp_file = '../Data/AMPs.fa'
    non_amp_file = '../Data/Non-AMPs.fa'

    sequences = []
    labels = []
    with open(amp_file, 'r') as amp:
        for line in amp:
            if not line.startswith('>'):
                sequences.append(line[:-1])
                labels.append(1)

    with open(non_amp_file, 'r') as amp:
        for line in amp:
            if not line.startswith('>'):
                sequences.append(line[:-1])
                labels.append(0)

    print(f'#of sequences for our paper: {len(sequences)}')
    return sequences,labels
        



class AminoAcidDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return sequence, label

