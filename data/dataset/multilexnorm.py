from data.dataset.utility import Indexer
from data.dataset.abstract_dataset import AbstractDataset


class MultilexnormDataset(AbstractDataset):
    def __init__(self, args, inputs, outputs):
        super().__init__(args, inputs, outputs)
        self.inputs = inputs
        self.outputs = outputs

        valid_indices = [[i for i, word in enumerate(sentence) if self.filter(word)] for sentence in inputs]
        self.indexer = Indexer(valid_indices)

    def __getitem__(self, index):
        sentence_index, word_index = self.indexer.get_indices(index)

        out = self.outputs[sentence_index][word_index]
        raw = self.inputs[sentence_index]

        raw = raw[:word_index] + ["<extra_id_0>", raw[word_index], "<extra_id_1>"] + raw[word_index+1:]
        raw = ' '.join(raw)

        return raw, out, sentence_index, word_index

    def __len__(self):
        return len(self.indexer)
