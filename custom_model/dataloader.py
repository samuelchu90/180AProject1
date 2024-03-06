from torch.utils.data import DataLoader, RandomSampler

def get_dataloader(dataset):
    sampler = RandomSampler(dataset)

    batch_size = 16
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    return dataloader