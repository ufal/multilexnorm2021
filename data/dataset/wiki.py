import torch

from data.noise.augmenter import Augmenter
from data.dataset.utility import RandomGenerator
from data.dataset.abstract_dataset import AbstractDataset


class WikiDataset(AbstractDataset):
    def __init__(self, args, inputs, outputs, language: str):
        super().__init__(args, inputs, outputs)

        path = {
            "en": "/lnet/work/people/samuel/wiki/en_processed_wiki.txt",
            "it": "/lnet/work/people/samuel/wiki/it_processed_wiki.txt",
            "sl": "/lnet/work/people/samuel/wiki/sl_processed_wiki.txt",
            "hr": "/lnet/work/people/samuel/wiki/hr_processed_wiki.txt",
            "da": "/lnet/work/people/samuel/wiki/da_processed_wiki.txt",
            "nl": "/lnet/work/people/samuel/wiki/nl_processed_wiki.txt",
            "sr": "/lnet/work/people/samuel/wiki/sr_romanized_wiki.txt",
            "id": "/lnet/work/people/samuel/wiki/id_processed_wiki.txt",
            "de": "/lnet/work/people/samuel/wiki/de_noneszett_wiki.txt",
            "tr": "/lnet/work/people/samuel/wiki/tr_processed_wiki.txt",
            "es": "/lnet/work/people/samuel/wiki/es_processed_wiki.txt",
        }[language]

        self.max_length = args.encoder_max_length

        with open(path) as f:
            self.sentences = [(sentence.lower() if args.lowercase else sentence)[:-1].split(' ') for sentence in f.readlines()]
            self.sentences = [s for s in self.sentences if len(' '.join(s)) >= 16 and len(' '.join(s)) <= self.max_length]
        self.augmenter = Augmenter(args, inputs, outputs, lowercase=args.lowercase)

#        self.n_total, self.n_diff = 0, 0
        print("\n\nDATASET TEST OUTPUT:\n")
        for i in range(100):
            raw, out, _, _ = self.__getitem__(i)
            if i < 100:
                print(raw)
                print(out)
#        print(self.n_diff / self.n_total * 100.0, flush=True)
#        exit()

    def __getitem__(self, sentence_index):
        random_generator = RandomGenerator(n_cached=32)

        if (
            sentence_index + 2 < len(self.sentences) and
            len(' '.join(self.sentences[sentence_index] + self.sentences[sentence_index+1] + self.sentences[sentence_index+2])) <= self.max_length and
            random_generator.pop() < 0.0678
        ):
            sentence = self.sentences[sentence_index] + self.sentences[sentence_index + 1] + self.sentences[sentence_index + 2]
        elif (
            sentence_index + 1 < len(self.sentences) and
            len(' '.join(self.sentences[sentence_index] + self.sentences[sentence_index+1])) <= self.max_length and
            random_generator.pop() < 0.273
        ):
            sentence = self.sentences[sentence_index] + self.sentences[sentence_index + 1]
        else:
            sentence = self.sentences[sentence_index]

        corrupted_sentence, gold_sentence = self.augmenter.augment(sentence, random_generator)
#        self.n_total += len(corrupted_sentence)
#        self.n_diff += sum([a != b for a, b in zip(corrupted_sentence, gold_sentence)])
        valid_indices = [i for i, word in enumerate(corrupted_sentence) if self.filter(word)]
        if len(valid_indices) == 0:
            word_index = 0
        else:
            word_index = valid_indices[torch.randint(low=0, high=len(valid_indices), size=(1,)).item()]

        raw = corrupted_sentence[:word_index] + ["<extra_id_0>", corrupted_sentence[word_index], "<extra_id_1>"] + corrupted_sentence[word_index+1:]
        raw = ' '.join([w for w in raw if w])

        out = gold_sentence[word_index]

        return raw, out, sentence_index, word_index

    def __len__(self):
        return len(self.sentences)
