import pytorch_lightning as pl
from transformers import T5ForConditionalGeneration
from utility.adafactor import Adafactor
import torch


class Model(pl.LightningModule):
    def __init__(self, args, dataset):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.tokenizer = self.dataset.tokenizer
        self.model = T5ForConditionalGeneration.from_pretrained(args.model.pretrained_lm)
        self.model.train()
        self.is_finetuning = False
        self.step = 1.0
        self.step_size = dataset.batch_size * args.trainer.n_gpus / args.trainer.total_batch_size

    def forward(self, batch):
        output = self.model(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            labels=batch["labels"], decoder_attention_mask=batch["decoder_attention_mask"]
        )
        loss = output.loss
        return loss

    def generate(self, batch):
        n_beams = self.args.model.n_beams
        sentence_ids, word_ids = batch["sentence_ids"], batch["word_ids"]

        outputs = self.model.generate(
            input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
            num_beams=n_beams, num_return_sequences=n_beams,
            repetition_penalty=1.0, length_penalty=1.0, max_length=32,
            output_scores=True, return_dict_in_generate=True
        )

        if n_beams > 1:
            scores = outputs.sequences_scores.cpu()
            scores = [scores[i*n_beams:(i+1)*n_beams] for i in range(len(sentence_ids))]
        else:
            scores = [[0.0] for i in range(len(sentence_ids))]

        outputs = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        outputs = [outputs[i*n_beams:(i+1)*n_beams] for i in range(len(sentence_ids))]

        out_dict = {
            "predictions": outputs,
            "scores": scores,
            "sentence_ids": sentence_ids,
            "word_ids": word_ids,
        }
        return out_dict

    def training_step(self, batch, _):
        self.step += self.step_size
        loss = self.forward(batch)

        logs = {"loss": loss, "real_step": self.step}
        self.log_dict({f"train/{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False)

        return loss

    def validation_step(self, batch, _):
        if not self.trainer.is_global_zero:
            return

        loss = self.forward(batch)
        out_dict = self.generate(batch)

        logs = {"loss": loss}
        self.log_dict({f"valid/{k}": v for k, v in logs.items()})

        return out_dict

    def train_dataloader(self):
        return self.dataset.get_train_dataloader(self.is_finetuning)

    def val_dataloader(self):
        return self.dataset.get_valid_dataloader()

    def configure_optimizers(self):
        if self.args.trainer.optimizer == "adafactor":
            self.optimizer = Adafactor(self.parameters(), lr=1e-3, relative_step=False, scale_parameter=False)
        else:
            self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)

        return self.optimizer
