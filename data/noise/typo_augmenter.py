from data.noise.utility import get_random_index, get_random_item


class TypoAugmenter:
    def __init__(self):
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"
        self.neighbors = {
            "a": "qwsz",
            "b": "vghn",
            "c": "xdfv",
            "d": "erfcxs",
            "e": "wsdfr34",
            "f": "rtgvcd",
            "g": "tyhbvf",
            "h": "yujnbg",
            "i": "ujklo89",
            "j": "uikmnh",
            "k": "iolmj",
            "l": "opk",
            "m": "njk",
            "n": "bhjm",
            "o": "ilp09",
            "p": "0lo",
            "q": "12wsa",
            "r": "45tfde",
            "s": "qwedxza",
            "t": "56ygfr",
            "u": "78ijhy",
            "v": "cfgb",
            "w": "qasde",
            "x": "zsdc",
            "y": "67uhgt",
            "z": "asx`",
        }
        for ch in self.alphabet:
            assert ch in self.neighbors

    def augment(self, word, p):
        random_float = p.pop()
        if random_float < 0.15:
            return self.repeat(word, p)
        if random_float < 0.4:
            return self.skip(word, p)
        if random_float < 0.5:
            return self.reverse(word, p)
        if random_float < 0.8:
            return self.change(word, p)
        return self.insert(word, p)

    def repeat(self, word, p):
        index = get_random_index(p.pop(), len(word))
        return word[:index] + word[index] + word[index] + word[index+1:]

    def skip(self, word, p):
        index = get_random_index(p.pop(), len(word))
        return word[:index] + word[index+1:]

    def reverse(self, word, p):
        index = get_random_index(p.pop(), len(word) - 1)
        return word[:index] + word[index + 1] + word[index] + word[index+2:]

    def change(self, word, p):
        index = get_random_index(p.pop(), len(word))
        if word[index].lower() not in self.alphabet:
            return word

        if word[index].lower() in self.alphabet and p.pop() < 0.6:
            character = get_random_item(self.neighbors[word[index].lower()], p.pop())
        else:
            character = get_random_item(self.alphabet, p.pop())

        if word[index].isupper():
            character = character.upper()

        return word[:index] + character + word[index+1:]

    def insert(self, word, p):
        index = get_random_index(p.pop(), len(word) + 1)

        random_float = p.pop()
        if random_float < 0.3 and word[max(index-1, 0)].lower() in self.alphabet:
            character = get_random_item(self.neighbors[word[max(index-1, 0)].lower()], p.pop())
            is_upper = word[max(index-1, 0)].isupper()
        if random_float < 0.3 and word[min(index, len(word)-1)].lower() in self.alphabet:
            character = get_random_item(self.neighbors[word[min(index, len(word)-1)].lower()], p.pop())
            is_upper = word[min(index, len(word)-1)].isupper()
        else:
            character = get_random_item(self.alphabet, p.pop())
            is_upper = word[min(index, len(word)-1)].isupper()

        if is_upper:
            character = character.upper()

        return word[:index] + character + word[index:]
