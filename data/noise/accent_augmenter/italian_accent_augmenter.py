class ItalianAccentAugmenter:
    def __init__(self, args):
        self.args = args

        self.accentate = ["à", "è", "é", "ì", "ò", "ù", "À", "È", "É", "Ì", "Ò", "Ù"]
        self.nonAccentate = ["a", "e", "e", "i", "o", "u", "A", "E", "E", "I", "O", "U"]

    def augment(self, word, p):
        if "é" in word:
            if p.pop() < self.args.replace_e_e_accents ** self.args.multiplier:
                word = word.replace("é", "è")
        elif "É" in word:
            if p.pop() < self.args.replace_e_e_accents ** self.args.multiplier:
                word = word.replace("É", "È")
        elif "è" in word:
            if p.pop() < self.args.replace_e_e_accents_reverse ** self.args.multiplier:
                word = word.replace("è", "é")
        elif "È" in word:
            if p.pop() < self.args.replace_e_e_accents_reverse ** self.args.multiplier:
                word = word.replace("È", "É")

        if "à" in word:
            if p.pop() < self.args.replace_a_a_accents ** self.args.multiplier:
                word = word.replace("à", "á")
        elif "À" in word:
            if p.pop() < self.args.replace_a_a_accents ** self.args.multiplier:
                word = word.replace("À", "Á")

        if p.pop() < self.args.remove_accents ** self.args.multiplier:
            word = self.deaccentate(word)

        if p.pop() < self.args.split_accents ** self.args.multiplier:
            word = self.split_accentate(word)

        return word

    def deaccentate(self, word):
        for source, target in zip(self.accentate, self.nonAccentate):
            word = word.replace(source, target)
        return word

    def split_accentate(self, word):
        for source, target in zip(self.accentate, self.nonAccentate):
            word = word.replace(source, target + "'")
        return word
