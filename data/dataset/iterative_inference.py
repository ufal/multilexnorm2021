import itertools
import math

from data.dataset.utility import Indexer
from data.dataset.abstract_dataset import AbstractDataset


class IterativeInferenceDataset(AbstractDataset):
    def __init__(self, args, inputs, previous_outputs):
        self.original_inputs = inputs
        self.inputs, self.lengths, self.indices = [], [], []

        for output_sent in previous_outputs:
            inputs, indices = [], []
            for i, output_words in enumerate(output_sent):
                if output_words == '':
                    continue
                output_words = output_words.split(' ')
                inputs += output_words
                indices += len(output_words) * [i]
            self.lengths.append(len(output_sent))
            self.inputs.append(inputs)
            self.indices.append(indices)

        valid_indices = [[i for i, word in enumerate(sentence)] for sentence in self.inputs]
        self.indexer = Indexer(valid_indices)

    def __getitem__(self, index):
        sentence_index, word_index = self.indexer.get_indices(index)

        raw = self.inputs[sentence_index]
        raw = raw[:word_index] + ["<extra_id_0>", raw[word_index], "<extra_id_1>"] + raw[word_index+1:]
        raw = ' '.join(raw)

        return raw, sentence_index, word_index

    def __len__(self):
        return len(self.indexer)

    def decode(self, predictions):
        out_predictions = []
        for indices, length, sentence in zip(self.indices, self.lengths, predictions):
            out_sentence = [[] for _ in range(length)]
            for index, words in zip(indices, sentence):
                out_sentence[index].append(words)
            for i in range(length):
                out_sentence[i] = self.combine(out_sentence[i])
            out_predictions.append(out_sentence)
        return out_predictions

    # the original token can be decoded into multiple words, combine them into word with spaces
    # the caveat is that each word is now made of a list of possible predictions
    def combine(self, words):
        if len(words) == 0:
            return [('', 0.0)]
        if len(words) == 1:
            return words[0]

        words = [w[:int(math.ceil(len(w) ** (1 / len(words))))] for w in words]
        words = [self.join(w) for w in itertools.product(*words)]
        return words

    # the same thing but on a lower level, now we'd like to combine single predictions
    def join(self, words):
        word = ' '.join([word[0] for word in words])
        logprob = sum([word[1] for word in words])
        return word, logprob
