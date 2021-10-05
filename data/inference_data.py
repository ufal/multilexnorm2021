from torch.utils.data import DataLoader
from data.abstract_data import AbstractData, open_dataset
from data.dataset.inference import InferenceDataset


class CollateFunctor:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, samples):
        inputs, sentence_indices, word_indices = map(list, zip(*samples))

        inputs = self.tokenizer(
            inputs, padding=True, truncation=False, pad_to_multiple_of=8,
            return_attention_mask=True, return_tensors='pt'
        )
        batch = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "word_ids": word_indices,
            "sentence_ids": sentence_indices
        }
        return batch


class InferenceData(AbstractData):
    def __init__(self, args):
        super().__init__(args)

        inputs = open_dataset(args.dataset.path, load_outputs=False)
        self.dataset = InferenceDataset(inputs)

        collate_fn = CollateFunctor(self.tokenizer)

        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.threads, collate_fn=collate_fn
        )

        print(f"number of tokens: {len(self.dataset)}", flush=True)
