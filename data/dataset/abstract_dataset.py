from torch.utils.data import Dataset


class AbstractDataset(Dataset):
    def __init__(self, args, inputs, outputs):
        if args.filter == "none":
            self.filter = lambda x: True
        elif args.filter == "alnum":
            def replace(word): return word.replace("'", "").replace("-", "").replace(" ", "")
            self.filter = lambda x: len(replace(x)) == 0 or replace(x).isalnum()
