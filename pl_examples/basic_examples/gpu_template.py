"""
Runs a model on a single node across multiple gpus.
"""
import os
from argparse import ArgumentParser

from pl_examples.models.lightning_template import LightningTemplateModel
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

seed_everything(234)


def main(args):
    """ Main training routine specific for this project. """
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = LightningTemplateModel(**vars(args))
    ctr = 0
    path = os.path.join(os.getcwd(), 'runs/lightning_logs/version_' + str(ctr) + '/checkpoints/epoch={epoch}')
    checkpoint_callback = ModelCheckpoint(
        save_top_k=1,
        filepath=path,
        verbose=True,
        monitor='val_loss',
        mode='min')

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=3,
        verbose=False,
        mode='min')

    trainer = Trainer(gpus=4, progress_bar_refresh_rate=0, max_epochs=10,
                         checkpoint_callback=checkpoint_callback,
                         distributed_backend="ddp")  # , distributed_backend="ddp"
    trainer.fit(model)

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    trainer = Trainer.from_argparse_args(args)

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    trainer.fit(model)


def run_cli():
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    root_dir = os.path.dirname(os.path.realpath(__file__))
    parent_parser = ArgumentParser(add_help=False)

    # each LightningModule defines arguments relevant to it
    parser = LightningTemplateModel.add_model_specific_args(parent_parser, root_dir)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(gpus=2)
    args = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)


if __name__ == '__main__':
    run_cli()
