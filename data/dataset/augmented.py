import torch

from data.noise.augmenter import Augmenter
from data.dataset.utility import Indexer, RandomGenerator
from data.dataset.abstract_dataset import AbstractDataset


class AugmentedDataset(AbstractDataset):
    def __init__(self, args, inputs, outputs, lowercase=True):
        self.inputs = inputs
        self.outputs = outputs
        self.indexer = Indexer([len(sentence) for sentence in inputs])
        self.augmenter = Augmenter(args, inputs, outputs, lowercase=lowercase)

        print("\n\nDATASET TEST OUTPUT:\n")
        randoms = torch.randint(0, len(self), (100,))
        for i in range(100):
            raw, out, _, _ = self.__getitem__(randoms[i].item())
            print(raw)
            print(out)
        print(flush=True)

    def __getitem__(self, index):
        sentence_index, word_index = self.indexer.get_indices(index)
        random_generator = RandomGenerator(n_cached=16)

        raw = self.inputs[sentence_index]
        out = self.outputs[sentence_index]

        if random_generator.pop() < 0.2:
            raw = self.inputs[sentence_index]
            raw = raw[:word_index] + ["<extra_id_0>", raw[word_index], "<extra_id_1>"] + raw[word_index+1:]
            raw = ' '.join(raw)

            out = self.outputs[sentence_index][word_index]

            return raw, out, sentence_index, word_index

        out = ' '.join(out).split(' ')
        corrupted_sentence, gold_sentence = self.augmenter.augment(out, random_generator)
        if word_index >= len(corrupted_sentence):
            word_index = torch.randint(low=0, high=len(corrupted_sentence), size=(1,)).item()

        raw = corrupted_sentence[:word_index] + ["<extra_id_0>", corrupted_sentence[word_index], "<extra_id_1>"] + corrupted_sentence[word_index+1:]
        raw = ' '.join([w for w in raw if w])

        out = gold_sentence[word_index]

        return raw, out, sentence_index, word_index

    def __len__(self):
        return len(self.indexer)
