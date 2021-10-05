import torch
from data.dataset.abstract_dataset import AbstractDataset


class MixedDataset(AbstractDataset):
    def __init__(self, dataset_a, dataset_b, dataset_b_portion: float):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b

        self.length = int(len(self.dataset_a) * (1.0 + dataset_b_portion))

    def __getitem__(self, index):
        if index < len(self.dataset_a):
            return self.dataset_a.__getitem__(index)

        index = torch.randint(len(self.dataset_b), size=(1,)).item()
        return self.dataset_b.__getitem__(index)

    def __len__(self):
        return self.length
