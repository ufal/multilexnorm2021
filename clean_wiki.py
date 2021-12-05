import stanza
import sys
import cyrtranslit

from data.noise.british_american_translator import BritishAmericanTranslator
from utility.twokenize import tokenizeRawTweetText


LANGUAGE = sys.argv[1]
INPUT_PATH = f"data/wiki/{LANGUAGE}_wiki.txt"
OUTPUT_PATH = f"data/wiki/{LANGUAGE}_processed_wiki.txt"
print(LANGUAGE, flush=True)

try:
    stanza.download(LANGUAGE, processors='tokenize', model_dir="/lnet/depot/samuel/cc")
    nlp = stanza.Pipeline(lang=LANGUAGE, processors='tokenize', model_dir="/lnet/depot/samuel/cc")
except:
    stanza.download(LANGUAGE, processors='tokenize,mwt', model_dir="/lnet/depot/samuel/cc")
    nlp = stanza.Pipeline(lang=LANGUAGE, processors='tokenize,mwt', model_dir="/lnet/depot/samuel/cc")

translator = BritishAmericanTranslator()
cache = set()


def romanize_sr(line):
    line = line.replace("-{", "")
    line = line.replace("- {", "")
    line = line.replace("}-", "")
    line = line.replace("} -", "")

    return cyrtranslit.to_latin(line)


def process_line(line):
    if LANGUAGE == "sr":
        line = romanize_sr(line)
    elif LANGUAGE == "de":
        line = line.replace("ß", "ss")

    line = line.strip()
    if len(line) <= 32 or line.isspace():
        return []

    if line.endswith(':'):
        return []

    try:
        sentences = nlp(line).sentences
    except:
        return []

    out_sentences = []
    for sentence in sentences:
        sentence = sentence.text
        if len(sentence) <= 32 or len(sentence) > 160:
            continue
        if sentence in cache:
            continue
        cache.add(sentence)
        tokens = tokenizeRawTweetText(sentence)

        if LANGUAGE == "en":
            tokens = translator.to_american(tokens)

        sentence = ' '.join(tokens) + '\n'
        out_sentences.append(sentence)

    return out_sentences


n_sentences = 0
with open(INPUT_PATH, encoding="utf8") as f:
    with open(OUTPUT_PATH, "w", encoding="utf8") as g:
        for line in f.readlines():
            sentences = process_line(line)
            n_sentences += len(sentences)
            for sentence in sentences:
                g.write(sentence)

            if n_sentences % 10000 == 0:
                print(n_sentences, flush=True)
