import torch
from torch.utils.data import DataLoader, Dataset

class PlotDataset(Dataset):
    def __init__(self, plots, labels, tokenizer):
        super().__init__()
        self.plots = plots.to_list()
        self.labels = labels.to_list()
        self.encodings = tokenizer(self.plots, truncation=True, padding=True)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        item = {key: torch.tensor(val[index]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[index],dtype=torch.long)
        return item

