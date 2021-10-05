import random


vocabulary = [
    ("is", ["not"]),
    ("are", ["not"]),
    ("was", ["not"]),
    ("were", ["not"]),
    ("have", ["not"]),
    ("has", ["not"]),
    ("had", ["not"]),
    ("would", ["not", "have"]),
    ("do", ["not"]),
    ("does", ["not"]),
    ("did", ["not"]),
    ("could", ["not", "have"]),
    ("should", ["not", "have"]),
    ("might", ["not", "have"]),
    ("must", ["not", "have"]),

    ("i", ["am", "will", "would", "have", "had"]),
    ("you", ["are", "will", "would", "have", "had"]),
    ("he", ["is", "will", "would", "has", "had"]),
    ("she", ["is", "will", "would", "has", "had"]),
    ("it", ["is", "will", "would", "has", "had"]),
    ("we", ["are", "will", "would", "have", "had"]),
    ("they", ["are", "will", "would", "have", "had"]),
    ("that", ["is", "will", "would", "has", "had"]),
    ("who", ["is", "are", "will", "would", "has", "have", "had"]),
    ("what", ["is", "are", "will", "would", "has", "have", "had"]),
    ("where", ["is", "are", "will", "would", "has", "have", "had"]),
    ("when", ["is", "are", "will", "would", "has", "have", "had"]),
    ("why", ["is", "are", "will", "would", "has", "have", "had"]),
    ("why", ["is", "are", "will", "would", "has", "have", "had"]),
    ("how", ["is", "are", "will", "would", "has", "have", "had"]),
]

contractions = {
    "am": "'m",
    "will": "'ll",
    "would": "'d",
    "have": "'ve",
    "had": "'d",
    "not": "n't",
    "are": "'re",
    "is": "'s",
    "has": "'s",
}

specials = [
    (["cannot"], "can't"),
    (["will", "not"], "won't"),
#   (["it", "is"], "'tis"),
]


class Contractor:
    def __init__(self, sentences):
        for i, (prefix, suffixes) in list(enumerate(vocabulary)):
            probs = []

            for suffix in suffixes:
                full_count, contraction_count = 1, 1  # a little bit of smoothing
                contraction = prefix + contractions[suffix]

                for sentence in sentences:
                    prefix_occurences = [i for i, w in enumerate(sentence[:-1]) if w == prefix]

                    if contraction in sentence:
                        contraction_count += 1
                    if any(sentence[i+1] == suffix for i in prefix_occurences):
                        full_count += 1

                probs.append((suffix, contraction_count / (full_count + contraction_count)))

            vocabulary[i] = (prefix, probs)

        for i, (full_words, contraction) in list(enumerate(specials)):
            if len(full_words) == 1:
                full_count, contraction_count = 1, 1  # a little bit of smoothing

                full_count = 1 + sum(full_words[0] in sentence for sentence in sentences)
                contraction_count = 1 + sum(contraction in sentence for sentence in sentences)

            else:
                prefix, suffix = full_words[0], full_words[1]
                full_count, contraction_count = 1, 1  # a little bit of smoothing

                for sentence in sentences:
                    prefix_occurences = [i for i, w in enumerate(sentence[:-1]) if w == prefix]

                    if contraction in sentence:
                        contraction_count += 1
                    if any(sentence[i+1] == suffix for i in prefix_occurences):
                        full_count += 1

            specials[i] = (full_words, (contraction, contraction_count / (full_count + contraction_count)))

        print(vocabulary)
        print()
        print(specials, flush=True)

    def augment(self, sentences):
        random.seed(42)
        return [self.augment_sentence(sentence.copy()) for sentence in sentences]

    def augment_sentence(self, sentence):
        for full_words, (contraction, p) in specials:
            if len(full_words) == 1:
                indices = [i for i, w in enumerate(sentence) if w == full_words[0]]
                for i in indices:
                    if random.random() < p:
                        sentence[i] = contraction
            else:
                indices = [i for i, w in enumerate(sentence[:-1]) if w == full_words[0] and sentence[i+1] == full_words[1]]
                offset = 0
                for i in indices:
                    if random.random() < p:
                        sentence = sentence[:i-offset] + [contraction] + sentence[i+2-offset:]
                        offset += 1

        for prefix, suffixes in vocabulary:
            prefix_occurences = [i for i, w in enumerate(sentence[:-1]) if w == prefix]

            offset = 0
            for suffix, p in suffixes:
                for i in prefix_occurences:
                    if sentence[i+1-offset] == suffix and random.random() < p:
                        sentence = sentence[:i-offset] + [prefix+contractions[suffix]] + sentence[i+2-offset:]
                        offset += 1

        return sentence
