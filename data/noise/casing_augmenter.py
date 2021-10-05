class CasingAugmenter:
    def __init__(self, args):
        self.args = args

    def augment(self, word, p):
        if word.islower():
            if p.pop() < self.args.first_upper_to_lower ** self.args.multiplier:
                return word[0].upper() + word[1:]
            if p.pop() < self.args.upper_to_lower ** self.args.multiplier:
                return word.upper()

        if word[0].isupper() and word[1:].islower():
            if p.pop() < self.args.lower_to_first_upper ** self.args.multiplier:
                return word.lower()
            if p.pop() < self.args.upper_to_first_upper ** self.args.multiplier:
                return word.upper()

        if word.isupper():
            if p.pop() < self.args.lower_to_upper ** self.args.multiplier:
                return word.lower()
            if p.pop() < self.args.first_upper_to_upper ** self.args.multiplier:
                return word[0] + word[1:].lower()

        return word
