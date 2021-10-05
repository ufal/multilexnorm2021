import re


class IndonesianRepetition:
    def __init__(self, args):
        self.args = args
        self.regex = re.compile(r'(\w*)(\w+)-\2(\w*)')

    def augment(self, word, random_generation):
        match = self.regex.findall(word)
        if len(match) == 0 or len(match[0]) != 3 or random_generation.pop() >= (1 / 3) ** self.args.multiplier:
            return word
        return f"{match[0][0]}{match[0][1]}2{match[0][2]}"

    def should_split(self, word, random_generation):
        match = self.regex.findall(word)
        if len(match) == 0 or len(match[0]) != 3 or random_generation.pop() >= 0.1711711 ** self.args.multiplier:
            return False
        return True

    def split(self, word):
        match = self.regex.findall(word)
        corrupted_words = [f"{match[0][0]}{match[0][1]}", f"{match[0][1]}{match[0][2]}"]
        gold_words = [word, ""]

        return corrupted_words, gold_words
