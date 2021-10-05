import pytorch_lightning as pl


class LRDecay(pl.Callback):
    def __init__(self, args):
        self.warmup_steps = args.warmup_steps
        self.base = args.peak_learning_rate
        self.decay_factor = self.base * self.warmup_steps ** 0.5
        self.finetune_learning_rate = args.finetune_learning_rate

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx):
        if pl_module.is_finetuning:
            lr = self.finetune_learning_rate
        else:
            step = int(pl_module.step)
            if step < 0:
                lr = 0.0
            elif step < self.warmup_steps:
                lr = self.base / self.warmup_steps * step
            else:
                lr = self.decay_factor * step ** -0.5

        for g in pl_module.optimizer.param_groups:
            g['lr'] = lr
        trainer.logger.agg_and_log_metrics({"learning_rate": lr})
