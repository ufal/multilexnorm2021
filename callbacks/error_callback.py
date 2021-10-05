import shutil
import pytorch_lightning as pl
from utility.error import Error
from data.abstract_data import open_dataset


class ErrorCallback(pl.Callback):
    def __init__(self, args, directory, dataset):
        super().__init__()
        self.error_processor = Error(dataset)
        self.language = args.dataset.language
        self.model_name = args.model.pretrained_lm.split('/')[-1]
        self.directory = directory
        self.best = float("-inf")
        self.mode = args.dataset.mode
        self.save_best = args.trainer.checkpoint == "best"
        self.seed = args.seed

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.is_global_zero:
            return

        directory = f"{self.directory}/test_predictions/{pl_module.current_epoch}"
        _, predictions = open_dataset(f"{directory}/outputs.txt")
        lai, accuracy, error, precision, recall, f1 = self.error_processor(predictions, directory)

        pl_module.log("valid/baseline_accuracy", lai)
        pl_module.log("valid/accuracy", accuracy)
        pl_module.log("valid/error", error)
        pl_module.log("valid/precision", precision)
        pl_module.log("valid/recall", recall)
        pl_module.log("valid/f1", f1)

        if not self.save_best or error > self.best:
            directory = f"checkpoints/{self.language}/{self.model_name}_{self.mode}_{self.seed}_{'best' if self.save_best else 'last'}"
#            pl_module.model.save_pretrained(directory)
        else:
            shutil.rmtree(directory)  # do not clutter the output directory

        if error > self.best:
            self.best = error
            pl_module.logger.experiment.summary["error"] = self.best
            pl_module.best = self.best
