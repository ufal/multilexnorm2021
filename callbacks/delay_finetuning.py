import pytorch_lightning as pl


class DelayFinetuning(pl.Callback):
    def __init__(self, args, dataset):
        super().__init__()
        self.dataset = dataset
        self.n_steps = args.n_steps
        self.is_pretraining = True

    def on_train_epoch_start(self, trainer, pl_module):
        if self.is_pretraining and pl_module.step >= self.n_steps:
            pl_module.is_finetuning = True
            trainer.reset_train_dataloader(pl_module)
            self.is_pretraining = False
