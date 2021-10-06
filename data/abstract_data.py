from transformers import T5TokenizerFast, ByT5Tokenizer


def open_dataset(path, load_outputs=True):
    with open(path) as f:
        sentences = f.read().split("\n\n")[:-1]
    sentences = [s.split('\n') for s in sentences]
    inputs = [[w.split('\t')[0] for w in s] for s in sentences]

    if not load_outputs:
        return inputs

    outputs = [[w.split('\t')[1] for w in s] for s in sentences]
    return inputs, outputs


class AbstractData:
    def __init__(self, args):
        tokenizer_factory = T5TokenizerFast if "mt5" in args.dataset.tokenizer else ByT5Tokenizer
        self.tokenizer = tokenizer_factory.from_pretrained(args.dataset.tokenizer)

        self.batch_size = args.dataset.batch_size
        self.threads = args.dataset.threads
        self.is_distributed = args.trainer.n_gpus > 1
        self.args = args.dataset
