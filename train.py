import datetime
import sys
import pytorch_lightning as pl
import os
import torch
from pytorch_lightning.utilities.apply_func import move_data_to_device

from model.model import Model
from callbacks.lr_decay import LRDecay
from callbacks.error_callback import ErrorCallback
from callbacks.output_callback import OutputCallback
from callbacks.delay_finetuning import DelayFinetuning
from callbacks.checkpoint_callback import CheckpointCallback
from config.params import Params
from data.training_data import TrainingData
from data.inference_data import InferenceData
from utility.output_assembler import OutputAssembler


if __name__ == "__main__":
    args = Params().load(sys.argv[1:])
    pl.seed_everything(args.seed)

    timestamp = f"{datetime.datetime.today():%m-%d-%y_%H-%M-%S}"
    directory = f"/lnet/work/people/samuel/outputs/{args.dataset.language}_{args.model.pretrained_lm.split('/')[-1]}_{timestamp}"
    predictions_directory = f"{directory}/test_predictions"
    os.mkdir(directory)
    os.mkdir(predictions_directory)

    if args.dataset.train_on_dev:
        name = f"{args.dataset.language}_{args.seed}_dev_{args.dataset.mode}"
        tags = [args.model.pretrained_lm, args.dataset.mode, args.dataset.language, "dev"]
    else:
        name = f"{args.model.pretrained_lm.split('/')[-1]}_{args.dataset.language}_{args.dataset.mode}"
        tags = [args.model.pretrained_lm, args.dataset.mode, args.dataset.language]

    wandb_logger = pl.loggers.WandbLogger(name=name, project="lex_norm", tags=tags)
    wandb_logger.log_hyperparams(args.state_dict())
    print(f"\nCONFIG:\n{args}")

    data = TrainingData(args)
    model = Model(args, data)

    trainer = pl.Trainer(
        accumulate_grad_batches=args.trainer.total_batch_size // args.dataset.batch_size, logger=wandb_logger,
        max_epochs=args.trainer.n_epochs, check_val_every_n_epoch=args.trainer.validate_each,
        callbacks=[
            LRDecay(args.trainer.lr_decay),
            OutputCallback(predictions_directory, args.trainer.output_callback, data.valid_set),
            ErrorCallback(args, directory, data.valid_set),
            DelayFinetuning(args.trainer.delay_finetuning, data),
            CheckpointCallback(args)
        ]
    )
    trainer.fit(model)


    # INFERENCE ON TEST

    def inference(args, model, input_data, output_dir):
        args.dataset.path = f"/home/samuel/personal_work_ms/w-nut-normalization/data/multilexnorm/test-eval/test/{input_data}"
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        data = InferenceData(args)
        assembler = OutputAssembler(output_dir, args.trainer.output_callback, data.dataset)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        for i, batch in enumerate(data.dataloader):
            batch = move_data_to_device(batch, device)
            output = model.generate(batch)
            assembler.step(output)
            print(f"{i} / {(len(data.dataset) + args.dataset.batch_size - 1) // args.dataset.batch_size}", flush=True)
        assembler.flush()

    def all_inference(args, model, mode, n_beams):
        if not os.path.isdir(f"ablation/{mode}"):
            os.mkdir(f"ablation/{mode}")

        args.model.n_beams = n_beams
        inference(args, model, f"intrinsic_evaluation/{args.dataset.language}/test.norm.masked", f"ablation/{mode}/{args.dataset.language}_{n_beams}")
        if args.dataset.language == "de":
            inference(args, model, f"extrinsic_evaluation/ud-de-tweede.test.norm.masked", f"ablation/{mode}/ud-de-tweede_{n_beams}")
        elif args.dataset.language == "en":
            inference(args, model, f"extrinsic_evaluation/ud-en-aae.test.norm.masked", f"ablation/{mode}/ud-en-aae_{n_beams}")
            inference(args, model, f"extrinsic_evaluation/ud-en-monoise.test.norm.masked", f"ablation/{mode}/ud-en-monoise_{n_beams}")
            inference(args, model, f"extrinsic_evaluation/ud-en-tweebank2.test.norm.masked", f"ablation/{mode}/ud-en-tweebank2_{n_beams}")
        elif args.dataset.language == "it":
            inference(args, model, f"extrinsic_evaluation/ud-it-postwita.test.norm.masked", f"ablation/{mode}/ud-it-postwita_{n_beams}")
            inference(args, model, f"extrinsic_evaluation/ud-it-twittiro.test.norm.masked", f"ablation/{mode}/ud-it-twittiro_{n_beams}")
        elif args.dataset.language == "tr":
            inference(args, model, f"extrinsic_evaluation/ud-tr-iwt151.test.norm.masked", f"ablation/{mode}/ud-tr-iwt151_{n_beams}")

    all_inference(args, model, "test_prediction_dir", 1)
