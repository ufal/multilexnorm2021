from data.noise.accent_augmenter.italian_accent_augmenter import ItalianAccentAugmenter
from data.noise.accent_augmenter.basic_accent_augmenter import BasicAccentAugmenter


class AccentAugmenterWrapper:
    def __init__(self, args):
        if args.language == "it":
            self.accent_augmenter = ItalianAccentAugmenter(args.noise_probs)
            return

        if args.language == "hr" or args.language == "sr":
            accents =     ["č", "ć", "đ", "š", "ž"]
            non_accents = ["c", "c", "d", "s", "z"]
        elif args.language == "nl":
            accents =     ['á', 'é', 'í', 'ó', 'ú', 'à', 'è', 'ë', 'ï', 'ö', 'ü', 'Á', 'É', 'Í', 'Ó', 'Ú', 'À', 'È', 'Ë', 'Ï', 'Ö', 'Ü']
            non_accents = ['a', 'e', 'i', 'o', 'u', 'a', 'e', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U', 'A', 'E', 'E', 'I', 'O', 'U']
        elif args.language == "de":
            accents =     ['ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü']
            non_accents = ['a', 'o', 'u', 'A', 'O', 'U']
        elif args.language == "sl":
            accents =     ["č", "š", "ž"]
            non_accents = ["c", "s", "z"]
        elif args.language == "tr":
            accents =     ['ı', 'â', 'î', 'û', 'ç', 'ğ', 'ş', 'I', 'Â', 'Î', 'Û', 'Ç', 'Ğ', 'Ş']
            non_accents = ['i', 'a', 'i', 'u', 'c', 'g', 's', 'I', 'A', 'I', 'U', 'C', 'G', 'S']
        elif args.language == "trde":
            accents =     ['ı', 'â', 'î', 'û', 'ç', 'ğ', 'ş', 'I', 'Â', 'Î', 'Û', 'Ç', 'Ğ', 'Ş', 'ä', 'ö', 'ü', 'Ä', 'Ö', 'Ü']
            non_accents = ['i', 'a', 'i', 'u', 'c', 'g', 's', 'I', 'A', 'I', 'U', 'C', 'G', 'S', 'a', 'o', 'u', 'A', 'O', 'U']
        elif args.language == "es":
            accents =     ['á', 'é', 'í', 'ó', 'ú', 'ñ', 'ü', '¡']
            non_accents = ['a', 'e', 'i', 'o', 'u', 'n', 'u', 'i']
        else:
            self.accent_augmenter = None
            return

        self.accent_augmenter = BasicAccentAugmenter(args.noise_probs, accents, non_accents)

    def augment(self, word, p):
        if self.accent_augmenter:
            return self.accent_augmenter.augment(word, p)
        return word
