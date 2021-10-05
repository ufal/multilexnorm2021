import math
import re
import itertools

from data.noise.vocabulary import Vocabulary
from data.noise.typo_augmenter import TypoAugmenter
from data.noise.british_american_translator import BritishAmericanTranslator
from data.noise.utility import skewed_probability, get_random_index
from data.noise.casing_augmenter import CasingAugmenter
from data.noise.accent_augmenter.accent_augmenter_wrapper import AccentAugmenterWrapper
from data.noise.indonesian_repetition import IndonesianRepetition


class Augmenter:
    def __init__(self, args, inputs, outputs, lowercase=False):
        self.dataset_vocabulary = Vocabulary(inputs, outputs)
        self.typo_augmenter = TypoAugmenter()
        self.casing_augmenter = CasingAugmenter(args.noise_probs)
        self.translator = BritishAmericanTranslator()
        self.indonesian_repetition = IndonesianRepetition(args.noise_probs)
        self.accent_augmenter = AccentAugmenterWrapper(args)
        self.lowercase = lowercase
        self.language = args.language
        self.vowels = ['U', 'Ò', 'è', 'E', 'È', 'Ö', 'e', 'Î', 'î', 'ä', 'ı', 'Û', 'Â', 'é', 'ó', 'Ì', 'A', 'I', 'i', 'Ù', 'à', 'ü', 'o', 'O', 'a', 'â', 'í', 'u', 'ú', 'ö', 'ò', 'Ü', 'û', 'ù', 'À', '¡', 'Ä', 'É', 'á', 'ì']
        self.args = args.noise_probs

        with open("verbs_list.txt", 'r') as f:
            self.english_verbs = set(f.read().split('\n'))

    def augment(self, words, random_generator):
        corrupted_sentence, gold_sentence = [], []
        words = words.copy()

        while len(words) > 0:
            current_words, words, gold_words = self.augment_word(words[0], words[1:], random_generator)

            gold_sentence += gold_words
            corrupted_sentence += current_words

        if self.language == "en" and random_generator.pop() < self.args.british_english:
            corrupted_sentence = self.translator.to_british(corrupted_sentence)

        return corrupted_sentence, gold_sentence

    def should_augment(self, p, name):
        if not hasattr(self.args, name):
            return False
        return p.pop() < getattr(self.args, name) ** self.args.multiplier

    def augment_word(self, word, next_words, p):
        gold_word = word
        corrupted_word = word

        checked_word = word.replace("'", "")
        if len(checked_word) == 0:
            return [word], next_words, [word]

        n_skip, corrupted_word = self.dataset_vocabulary.suggest([corrupted_word] + next_words, p.pop())
        gold_word = ' '.join([gold_word] + next_words[:n_skip])
        next_words = next_words[n_skip:]

        if self.language == "iden":
            corrupted_word = self.indonesian_repetition.augment(corrupted_word, p)

        if (self.language == "de" or self.language == "trde") and len(next_words) > 0 and next_words[0] == "es" and self.should_augment(p, "replaced_es"):
            gold_word = f"{corrupted_word} es"
            next_words = next_words[1:]

            if corrupted_word.endswith('e'):
                corrupted_word = corrupted_word[:-1] + 's'
            else:
                corrupted_word = corrupted_word + 's'

        if len(corrupted_word) > 0 and self.should_augment(p, "typo"):
            corrupted_word = self.typo_augmenter.augment(corrupted_word, p)

        if "'" in corrupted_word and len(corrupted_word) > 1 and self.should_augment(p, "missing_apostrophe"):
            corrupted_word = corrupted_word.replace("'", "")

        if len(corrupted_word) > 0 and not self.lowercase:
            corrupted_word = self.casing_augmenter.augment(corrupted_word, p)

        corrupted_word = self.accent_augmenter.augment(corrupted_word, p)

        if self.should_augment(p, "missing_vowels"):
            corrupted_word = ''.join([c for c in corrupted_word if c not in self.vowels])

        if self.should_augment(p, "remove_repeated"):
            corrupted_word = ''.join(c[0] for c in itertools.groupby(corrupted_word))

        if self.language == "iden" and corrupted_word in self.english_verbs and self.should_augment(p, "add_nge_prefix"):
            corrupted_word = "nge" + corrupted_word

        if self.language == "iden" and corrupted_word == "the" and len(next_words) > 0 and self.should_augment(p, "add_nya_suffix"):
            corrupted_word = next_words[0] + "nya"
            gold_word = f"the {next_words[0]}"
            next_words = next_words[1:]

        if "for" in corrupted_word and corrupted_word != "for" and self.should_augment(p, "replaced_for"):
            corrupted_word = corrupted_word.replace("for", "4")

        if "to" in corrupted_word and corrupted_word != "to" and self.should_augment(p, "replaced_to"):
            corrupted_word = corrupted_word.replace("to", "2")

        if "one" in corrupted_word and corrupted_word != "one" and self.should_augment(p, "replaced_one"):
            corrupted_word = corrupted_word.replace("one", "1")

        if "ß" in corrupted_word and self.should_augment(p, "replaced_eszett_ss"):
            corrupted_word = corrupted_word.replace("ß", "ss")
        if "ß" in corrupted_word and self.should_augment(p, "replaced_eszett_s"):
            corrupted_word = corrupted_word.replace("ß", "s")

        if "th" in corrupted_word and self.should_augment(p, "replaced_th"):
            corrupted_word = corrupted_word.replace("th", "d")

        if "ck" in corrupted_word and self.should_augment(p, "replaced_ck"):
            corrupted_word = corrupted_word.replace("ck", "k")

        if "ch" in corrupted_word and self.should_augment(p, "replaced_ch"):
            corrupted_word = corrupted_word.replace("ch", "k")
        if "Ch" in corrupted_word and self.should_augment(p, "replaced_ch"):
            corrupted_word = corrupted_word.replace("Ch", "K")
        if "CH" in corrupted_word and self.should_augment(p, "replaced_ch"):
            corrupted_word = corrupted_word.replace("CH", "K")

        if corrupted_word.endswith("er") and self.should_augment(p, "replaced_er"):
            corrupted_word = corrupted_word[:-2]
        if corrupted_word.endswith("ER") and self.should_augment(p, "replaced_er"):
            corrupted_word = corrupted_word[:-2]

        if corrupted_word.endswith("e") and self.should_augment(p, "replaced_e"):
            corrupted_word = corrupted_word[:-1]
        if corrupted_word.endswith("E") and self.should_augment(p, "replaced_e"):
            corrupted_word = corrupted_word[:-1]

        if corrupted_word.endswith("ao") and self.should_augment(p, "replaced_ao_o"):
            corrupted_word = corrupted_word[:-2] + "o"
        if corrupted_word.endswith("ao") and self.should_augment(p, "replaced_ao_a"):
            corrupted_word = corrupted_word[:-1]

        if "ue" in corrupted_word and self.should_augment(p, "replaced_ue_e"):
            corrupted_word = corrupted_word.replace("ue", "e")
        if "Ue" in corrupted_word and self.should_augment(p, "replaced_ue_e"):
            corrupted_word = corrupted_word.replace("Ue", "E")
        if "UE" in corrupted_word and self.should_augment(p, "replaced_ue_e"):
            corrupted_word = corrupted_word.replace("UE", "E")

        if "ch" in corrupted_word and self.should_augment(p, "replaced_ch_x"):
            corrupted_word = corrupted_word.replace("ch", "x")
        if "Ch" in corrupted_word and self.should_augment(p, "replaced_ch_x"):
            corrupted_word = corrupted_word.replace("Ch", "X")
        if "CH" in corrupted_word and self.should_augment(p, "replaced_ch_x"):
            corrupted_word = corrupted_word.replace("CH", "X")

        if "y" in corrupted_word and self.should_augment(p, "replaced_y_i"):
            corrupted_word = corrupted_word.replace("y", "i")
        if "Y" in corrupted_word and self.should_augment(p, "replaced_y_i"):
            corrupted_word = corrupted_word.replace("Y", "I")

        if "_" in corrupted_word and self.should_augment(p, "replaced_underscore"):
            corrupted_word = corrupted_word.replace("_", "")

        if "qu" in corrupted_word and self.should_augment(p, "replaced_qu_k"):
            corrupted_word = corrupted_word.replace("qu", "k")
        if "Qu" in corrupted_word and self.should_augment(p, "replaced_qu_k"):
            corrupted_word = corrupted_word.replace("Qu", "K")
        if "QU" in corrupted_word and self.should_augment(p, "replaced_qu_k"):
            corrupted_word = corrupted_word.replace("QU", "K")

        if "đ" in corrupted_word and self.should_augment(p, "replaced_d_dj"):
            corrupted_word = corrupted_word.replace("đ", "dj")

        if "š" in corrupted_word and self.should_augment(p, "replaced_s_sh"):
            corrupted_word = corrupted_word.replace("š", "sh")

        if "č" in corrupted_word and self.should_augment(p, "replaced_c_ch"):
            corrupted_word = corrupted_word.replace("č", "ch")

        if "c" in corrupted_word and self.should_augment(p, "replaced_c_k"):
            corrupted_word = corrupted_word.replace("c", "k")
        if "C" in corrupted_word and self.should_augment(p, "replaced_c_k"):
            corrupted_word = corrupted_word.replace("C", "K")

        if "z" in corrupted_word and self.should_augment(p, "replaced_z_s"):
            corrupted_word = corrupted_word.replace("z", "s")
        if "Z" in corrupted_word and self.should_augment(p, "replaced_z_s"):
            corrupted_word = corrupted_word.replace("Z", "S")

        if corrupted_word.endswith("'") and self.should_augment(p, "replaced_apostrophe"):
            corrupted_word = corrupted_word[:-1]

        if corrupted_word.endswith("s") and self.should_augment(p, "replaced_s"):
            corrupted_word = corrupted_word[:-1] + "z"

        if "ou" in corrupted_word and self.should_augment(p, "replaced_ou"):
            corrupted_word = corrupted_word.replace("ou", "u")

        if "u" in corrupted_word and self.should_augment(p, "replaced_u_a"):
            corrupted_word = corrupted_word.replace("u", "a")

        if "u" in corrupted_word and self.should_augment(p, "replaced_u_oo"):
            corrupted_word = corrupted_word.replace("u", "oo")

        if "th" in corrupted_word and self.should_augment(p, "replaced_th_f"):
            corrupted_word = corrupted_word.replace("th", "f")

        if corrupted_word.endswith("kke") and self.should_augment(p, "replaced_kke_k"):
            corrupted_word = corrupted_word[:-2]
        if corrupted_word.endswith("KKE") and self.should_augment(p, "replaced_kke_k"):
            corrupted_word = corrupted_word[:-2]

        if corrupted_word.endswith("er") and self.should_augment(p, "replaced_er_a"):
            corrupted_word = corrupted_word[:-2] + "a"

        if corrupted_word.endswith("ed") and self.should_augment(p, "replaced_ed"):
            corrupted_word = corrupted_word[:-2] + "t"

        if corrupted_word.endswith("ing") and self.should_augment(p, "replaced_ing"):
            corrupted_word = corrupted_word[:-1]

        if corrupted_word.endswith("aj") and self.should_augment(p, "replaced_aj"):
            corrupted_word = corrupted_word[:-2] + "ej"

        if corrupted_word.endswith("o") and self.should_augment(p, "replaced_o"):
            corrupted_word = corrupted_word[:-1]

        if corrupted_word.endswith("k") and self.should_augment(p, "replaced_k"):
            corrupted_word = corrupted_word[:-1]

        if corrupted_word.endswith("o") and self.should_augment(p, "replaced_o_a"):
            corrupted_word = corrupted_word[:-1] + "a"

        if corrupted_word.endswith("i") and self.should_augment(p, "replaced_end_i"):
            corrupted_word = corrupted_word[:-1]

        if "v" in corrupted_word and self.should_augment(p, "replaced_v_u"):
            corrupted_word = corrupted_word.replace("v", "u")

        if "i" in corrupted_word and self.should_augment(p, "replaced_i"):
            corrupted_word = corrupted_word.replace("i", "")

        if "-" in corrupted_word and self.should_augment(p, "replaced_hyphen"):
            corrupted_word = corrupted_word.replace("-", "")

        if corrupted_word.endswith("at") and self.should_augment(p, "replaced_at_a"):
            corrupted_word = corrupted_word[:-1]
        if corrupted_word.endswith("AT") and self.should_augment(p, "replaced_at_a"):
            corrupted_word = corrupted_word[:-1]

        if corrupted_word.endswith("en") and self.should_augment(p, "replaced_en_e"):
            corrupted_word = corrupted_word[:-1]
        if corrupted_word.endswith("EN") and self.should_augment(p, "replaced_en_e"):
            corrupted_word = corrupted_word[:-1]

        if corrupted_word.endswith("en") and self.should_augment(p, "replaced_en_n"):
            corrupted_word = corrupted_word[:-2] + 'n'
        if corrupted_word.endswith("EN") and self.should_augment(p, "replaced_en_n"):
            corrupted_word = corrupted_word[:-2] + 'N'

        if corrupted_word.startswith("h") and self.should_augment(p, "replaced_start_h"):
            corrupted_word = corrupted_word[1:]

        if corrupted_word.startswith("s") and self.should_augment(p, "replaced_s"):
            corrupted_word = corrupted_word[1:]

        if corrupted_word.startswith("m") and self.should_augment(p, "replaced_m"):
            corrupted_word = corrupted_word[1:]

        if corrupted_word.startswith("be") and self.should_augment(p, "replaced_be"):
            corrupted_word = corrupted_word[2:]

        if corrupted_word.startswith("me") and self.should_augment(p, "replaced_me"):
            corrupted_word = corrupted_word[2:]

        if corrupted_word.endswith("h") and self.should_augment(p, "replaced_h"):
            corrupted_word = corrupted_word[:-1]
        if corrupted_word.endswith("H") and self.should_augment(p, "replaced_H"):
            corrupted_word = corrupted_word[:-1]

        if corrupted_word.endswith("å") and self.should_augment(p, "replaced_a_aa"):
            corrupted_word = corrupted_word.replace("å", "aa")
        if corrupted_word.endswith("Å") and self.should_augment(p, "replaced_a_aa"):
            corrupted_word = corrupted_word.replace("Å", "AA")

        if corrupted_word.endswith("en") and self.should_augment(p, "replaced_en_n"):
            corrupted_word = corrupted_word.replace("en", "n")
        if corrupted_word.endswith("EN") and self.should_augment(p, "replaced_en_n"):
            corrupted_word = corrupted_word.replace("EN", "N")

        if corrupted_word.endswith("i") and self.should_augment(p, "replaced_end_i"):
            corrupted_word = corrupted_word[:-1]

        if corrupted_word.endswith("a") and self.should_augment(p, "replaced_a"):
            corrupted_word = corrupted_word[:-1]

        if corrupted_word.endswith("t") and self.should_augment(p, "replaced_t"):
            corrupted_word = corrupted_word[:-1]
        if corrupted_word.endswith("T") and self.should_augment(p, "replaced_T"):
            corrupted_word = corrupted_word[:-1]

        if corrupted_word.endswith("d") and self.should_augment(p, "replaced_d"):
            corrupted_word = corrupted_word[:-1]
        if corrupted_word.endswith("D") and self.should_augment(p, "replaced_D"):
            corrupted_word = corrupted_word[:-1]

        if "ü" in corrupted_word and self.should_augment(p, "replaced_u_ue"):
            corrupted_word = corrupted_word.replace("ü", "ue")
        if "Ü" in corrupted_word and self.should_augment(p, "replaced_u_ue"):
            corrupted_word = corrupted_word.replace("Ü", "UE")

        if "ä" in corrupted_word and self.should_augment(p, "replaced_a_ae"):
            corrupted_word = corrupted_word.replace("ä", "ae")
        if "Ä" in corrupted_word and self.should_augment(p, "replaced_a_ae"):
            corrupted_word = corrupted_word.replace("Ä", "AE")

        if "ö" in corrupted_word and self.should_augment(p, "replaced_o_oe"):
            corrupted_word = corrupted_word.replace("ö", "oe")
        if "Ö" in corrupted_word and self.should_augment(p, "replaced_o_oe"):
            corrupted_word = corrupted_word.replace("Ö", "OE")

        if "ss" in corrupted_word and self.should_augment(p, "replaced_ss_s"):
            corrupted_word = corrupted_word.replace("ss", "s")
        if "Ss" in corrupted_word and self.should_augment(p, "replaced_ss_s"):
            corrupted_word = corrupted_word.replace("Ss", "S")
        if "SS" in corrupted_word and self.should_augment(p, "replaced_ss_s"):
            corrupted_word = corrupted_word.replace("SS", "S")

        if corrupted_word.startswith("da") and self.should_augment(p, "replaced_da_d"):
            corrupted_word = "d" + corrupted_word[2:]
        if corrupted_word.startswith("Da") and self.should_augment(p, "replaced_da_d"):
            corrupted_word = "D" + corrupted_word[2:]
        if corrupted_word.startswith("DA") and self.should_augment(p, "replaced_da_d"):
            corrupted_word = "D" + corrupted_word[2:]

        if corrupted_word.startswith("ge") and self.should_augment(p, "replaced_ge_g"):
            corrupted_word = "g" + corrupted_word[2:]
        if corrupted_word.startswith("Ge") and self.should_augment(p, "replaced_ge_g"):
            corrupted_word = "G" + corrupted_word[2:]
        if corrupted_word.startswith("GE") and self.should_augment(p, "replaced_ge_g"):
            corrupted_word = "G" + corrupted_word[2:]

        if corrupted_word.startswith("i") and self.should_augment(p, "replaced_start_i"):
            corrupted_word = corrupted_word[1:]

        if len(corrupted_word) > 0 and self.should_augment(p, "repeated_letters"):
            n_repeats = int(math.floor(skewed_probability(p.pop()) * 10.0)) + 1
            index = get_random_index(p.pop(), len(corrupted_word))
            corrupted_word = corrupted_word[:index] + ''.join(corrupted_word[index] * n_repeats) + corrupted_word[index+1:]

        if len(corrupted_word) > 3 and self.should_augment(p, "prefix"):
            index = get_random_index(p.pop(), len(corrupted_word) - 3) + 1

            if corrupted_word.endswith("s") and p.pop() < 0.8:
                corrupted_word = corrupted_word[:index] + "s"
            elif corrupted_word.endswith("ed") and p.pop() < 0.5:
                corrupted_word = corrupted_word[:index] + "ed"
            else:
                corrupted_word = corrupted_word[:index]

        if self.should_augment(p, "joined_words"):
            n = get_random_index(skewed_probability(p.pop()), 3) + 1
            gold_word = ' '.join([corrupted_word] + next_words[:n])
            corrupted_word = ''.join([corrupted_word] + next_words[:n])
            next_words = next_words[n:]

        if len(corrupted_word) > 2 and self.should_augment(p, "split_words"):
            index = get_random_index(p.pop(), len(corrupted_word) - 1) + 1
            corrupted_words = [corrupted_word[:index], corrupted_word[index:]]
            gold_words = [gold_word, ""]
        elif len(corrupted_word) > 2 and self.should_augment(p, "split_into_letters"):
            corrupted_words = [letter for letter in corrupted_word]
            gold_words = [gold_word] + ["" for _ in range(len(corrupted_word) - 1)]
        elif self.language == "iden" and self.indonesian_repetition.should_split(corrupted_word, p):
            corrupted_words, gold_words = self.indonesian_repetition.split(corrupted_word)
        else:
            corrupted_words = [corrupted_word]
            gold_words = [gold_word]

        return corrupted_words, next_words, gold_words
