import pytorch_lightning as pl
from utility.output_assembler import OutputAssembler


class OutputCallback(pl.Callback):
    def __init__(self, directory, args, dataset):
        super().__init__()
        self.directory = directory
        self.args = args
        self.dataset = dataset

    def on_validation_epoch_start(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        directory = f"{self.directory}/{pl_module.current_epoch}"
        self.assembler = OutputAssembler(directory, self.args, self.dataset)

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        self.assembler.flush()
        del self.assembler

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not trainer.is_global_zero:
            return

        self.assembler.step(outputs)
