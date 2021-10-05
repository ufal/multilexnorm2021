class BasicAccentAugmenter:
    def __init__(self, args, accents, non_accents):
        self.args = args
        self.accents = accents
        self.non_accents = non_accents

    def augment(self, word, p):
        if p.pop() < self.args.remove_accents ** self.args.multiplier:
            word = self.remove_accents(word)

        return word

    def remove_accents(self, word):
        for source, target in zip(self.accents, self.non_accents):
            word = word.replace(source, target)
        return word
