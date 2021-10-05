from collections import deque
import torch


class RandomGenerator:
    def __init__(self, n_cached=16):
        self.n_cached = 16
        self.fill()

    def fill(self):
        self.stack = deque(torch.rand(self.n_cached).tolist())

    def pop(self):
        if len(self.stack) == 0:
            self.fill()
        return self.stack.pop()


class Indexer:
    def __init__(self, valid_indices):
        self.valid_indices = valid_indices
        lengths = [len(sentence_indices) for sentence_indices in self.valid_indices]
        self.cumsum = torch.LongTensor([0] + lengths).cumsum(dim=0)

    def get_indices(self, index):
        sentence_index = torch.searchsorted(self.cumsum, index, right=True).item() - 1
        word_index = index - self.cumsum[sentence_index]
        word_index = self.valid_indices[sentence_index][word_index]

        return sentence_index, word_index

    def __len__(self):
        return self.cumsum[-1].item()
