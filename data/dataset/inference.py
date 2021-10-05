from data.dataset.utility import Indexer
from data.dataset.abstract_dataset import AbstractDataset


class InferenceDataset(AbstractDataset):
    def __init__(self, inputs):
        self.inputs = inputs

        valid_indices = [[i for i, word in enumerate(sentence)] for sentence in inputs]
        self.indexer = Indexer(valid_indices)
        self.sorted_indices = sorted(list(range(len(self))), key=lambda x: (len(self.inputs[self.indexer.get_indices(x)[0]]), len(self.inputs[self.indexer.get_indices(x)[0]][self.indexer.get_indices(x)[1]])))

    def __getitem__(self, index):
        index = self.sorted_indices[index]
        sentence_index, word_index = self.indexer.get_indices(index)

        raw = self.inputs[sentence_index]
        raw = raw[:word_index] + ["<extra_id_0>", raw[word_index], "<extra_id_1>"] + raw[word_index+1:]
        raw = ' '.join(raw)

        return raw, sentence_index, word_index

    def __len__(self):
        return len(self.indexer)
