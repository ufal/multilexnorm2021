import torch
from data.noise.utility import get_random_index


class Vocabulary:
    def __init__(self, inputs, outputs):
        inputs_flat = [w for s in inputs for w in s]
        outputs_flat = [w for s in outputs for w in s]
        self.dictionary = {}
        for i, os in zip(inputs_flat, outputs_flat):
            if i == os:
                continue
            os = os.split(" ")
            d = self.dictionary
            for o in os:
                d = d.setdefault(o, {})
            assert len(i.split(' ')) == 1
            d.setdefault("<<values>>", [None]).append(i)

        for input_words, output_words in zip(inputs, outputs):
            for start in range(len(input_words)):
                words = []
                for end in range(start, len(input_words)):
                    if input_words[end] == output_words[end]:
                        words.append(input_words[end])
                    else:
                        break
                if len(words) == 0:
                    continue

                lengths = self.occurence_lengths(words)
                if len(lengths) == 0:
                    continue

                n = 0
                for i, length in enumerate(lengths):
                    if length > 0:
                        n = i + 1

                if n == 0:
                    continue

                d = self.dictionary
                for i in range(n):
                    if words[i] not in d:
                        break
                    d = d[words[i]]

                if "<<values>>" in d:
                    d["<<values>>"].append(None)

    def occurence_lengths(self, words):
        d = self.dictionary
        lengths = []
        for word in words:
            if word not in d:
                return lengths
            d = d[word]
            if "<<values>>" not in d:
                lengths.append(0)
            else:
                lengths.append(len(d["<<values>>"]))

        return lengths

    def suggest(self, words, random_float):
        lengths = self.occurence_lengths(words)
        if len(lengths) == 0:
            return 0, words[0]

        cumsum = torch.LongTensor([0] + lengths).cumsum(dim=0)
        if cumsum[-1] == 0:
            return 0, words[0]

        index = get_random_index(random_float, cumsum[-1])
        out_index = torch.searchsorted(cumsum, index, right=True).item() - 1
        in_index = index - cumsum[out_index]

        d = self.dictionary
        for word in words[:out_index+1]:
            d = d[word]

        if d["<<values>>"][in_index] is None:
            return 0, words[0]
        return out_index, d["<<values>>"][in_index]
