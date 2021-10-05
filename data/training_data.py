import random
import os
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from data.dataset.multilexnorm import MultilexnormDataset
from data.dataset.wiki import WikiDataset
from data.dataset.augmented import AugmentedDataset
from data.dataset.mixed import MixedDataset
from data.abstract_data import AbstractData, open_dataset


class CollateFunctor:
    def __init__(self, tokenizer, encoder_max_length, decoder_max_length):
        self.tokenizer = tokenizer
        self.encoder_max_length = encoder_max_length
        self.decoder_max_length = decoder_max_length

    def __call__(self, samples):
        inputs, outputs, sentence_indices, word_indices = map(list, zip(*samples))

        inputs = self.tokenizer(
            inputs, padding=True, truncation=True, pad_to_multiple_of=8,
            max_length=self.encoder_max_length, return_attention_mask=True, return_tensors='pt'
        )
        outputs = self.tokenizer(
            outputs, padding=True, truncation=True, pad_to_multiple_of=8,
            max_length=self.decoder_max_length, return_attention_mask=True, return_tensors='pt'
        )

        batch = {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": outputs.input_ids,
            "decoder_attention_mask": outputs.attention_mask,
            "word_ids": word_indices,
            "sentence_ids": sentence_indices
        }
        batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100  # used to mask the loss in T5
        return batch


class TrainingData(AbstractData):
    def __init__(self, args):
        super().__init__(args)
        (train_inputs, train_outputs), (valid_inputs, valid_outputs) = self.load(args)

        def get_wiki_dataset(args, inputs, outputs, language):
            if len(language) == 4:
                dataset_a = WikiDataset(args, inputs, outputs, language[:2])
                dataset_b = WikiDataset(args, inputs, outputs, language[2:])
                return MixedDataset(dataset_a, dataset_b, dataset_b_portion=args.second_language_portion)
            else:
                return WikiDataset(args, inputs, outputs, language)

        if args.dataset.mode == "augmented":
            self.augmented_train_set = AugmentedDataset(args.dataset, train_inputs, train_outputs)
            self.lexnorm_train_set = MultilexnormDataset(args.dataset, train_inputs, train_outputs)
        elif args.dataset.mode == "baseline" or args.dataset.mode == "finetune":
            self.augmented_train_set = MultilexnormDataset(args.dataset, train_inputs, train_outputs)
            self.lexnorm_train_set = MultilexnormDataset(args.dataset, train_inputs, train_outputs)
        elif args.dataset.mode == "wiki":
            self.augmented_train_set = get_wiki_dataset(args.dataset, train_inputs, train_outputs, args.dataset.language)
            self.lexnorm_train_set = MultilexnormDataset(args.dataset, train_inputs, train_outputs)
        elif args.dataset.mode == "mixed":
            multilexnorm_train_set = MultilexnormDataset(args.dataset, train_inputs, train_outputs)
            wiki_train_set = get_wiki_dataset(args.dataset, train_inputs, train_outputs, args.dataset.language)
            self.augmented_train_set = MixedDataset(multilexnorm_train_set, wiki_train_set, dataset_b_portion=args.dataset.wiki_portion)
            self.lexnorm_train_set = self.augmented_train_set

        self.valid_set = MultilexnormDataset(args.dataset, valid_inputs, valid_outputs)

        print(f"number of wiki train sentences: {len(self.augmented_train_set)}")
        print(f"number of lexnorm train sentences: {len(self.lexnorm_train_set)}")
        print(f"number of lexnorm valid sentences: {len(self.valid_set)}")

    def load(self, args):
        train_path = f"data/multilexnorm/data/{args.dataset.language}/train.norm"
        valid_path = f"data/multilexnorm/data/{args.dataset.language}/dev.norm"

        train_dataset = open_dataset(train_path)

        if os.path.exists(valid_path):
            valid_dataset = open_dataset(valid_path)

            if args.dataset.train_on_dev:
                valid_dataset = list(zip(*valid_dataset))
                random.Random(args.seed).shuffle(valid_dataset)
                inputs = [s[0] for s in valid_dataset] + train_dataset[0]
                outputs = [s[1] for s in valid_dataset] + train_dataset[1]

                valid_dataset = inputs[:len(inputs) // 33], outputs[:len(outputs) // 33]
                train_dataset = inputs[len(inputs) // 33:], outputs[len(outputs) // 33:]

        else:
            dataset = list(zip(*train_dataset))
            random.Random(args.seed).shuffle(dataset)
            inputs = [s[0] for s in dataset]
            outputs = [s[1] for s in dataset]

            portion = 33 if args.dataset.train_on_dev else 10

            valid_dataset = inputs[:len(inputs) // portion], outputs[:len(outputs) // portion]
            train_dataset = inputs[len(inputs) // portion:], outputs[len(outputs) // portion:]

        return train_dataset, valid_dataset

    def get_train_dataloader(self, is_finetuning):
        collate_fn = CollateFunctor(self.tokenizer, self.args.encoder_max_length, self.args.decoder_max_length)
        if is_finetuning:
            sampler = DistributedSampler(self.lexnorm_train_set, shuffle=True) if self.is_distributed else None
            return DataLoader(
                self.lexnorm_train_set, batch_size=self.batch_size, shuffle=sampler is None, drop_last=True,
                num_workers=self.threads, sampler=sampler, collate_fn=collate_fn
            )
        else:
            sampler = DistributedSampler(self.augmented_train_set, shuffle=True) if self.is_distributed else None
            return DataLoader(
                self.augmented_train_set, batch_size=self.batch_size, shuffle=sampler is None, drop_last=True,
                num_workers=self.threads, sampler=sampler, collate_fn=collate_fn
            )

    def get_valid_dataloader(self):
        collate_fn = CollateFunctor(self.tokenizer, None, None)
        return DataLoader(
            self.valid_set, batch_size=self.batch_size, shuffle=False, num_workers=self.threads, collate_fn=collate_fn
        )
