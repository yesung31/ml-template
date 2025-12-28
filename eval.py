import argparse
import os
import sys

import pytorch_lightning as pl

import data
import models
from utils.helpers import instantiate, load_config_from_ckpt


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("ckpt_path", type=str, help="Path to the checkpoint file (.ckpt)")
    args = parser.parse_args()

    if not os.path.exists(args.ckpt_path):
        print(f"Checkpoint not found: {args.ckpt_path}")
        sys.exit(1)

    cfg = load_config_from_ckpt(args.ckpt_path)

    pl.seed_everything(cfg.seed)

    # Dynamic loading
    ModelClass = getattr(models, cfg.model.name)
    DataClass = getattr(data, cfg.data.name)

    print(f"Model Class: {ModelClass.__name__}")
    print(f"Data Module Class: {DataClass.__name__}")

    # Initialize DataModule
    dm = instantiate(DataClass, cfg.data)

    # Load Model from Checkpoint
    print(f"Loading model from {args.ckpt_path}")
    model = ModelClass.load_from_checkpoint(args.ckpt_path)

    # Trainer for testing
    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        logger=False,  # No logging during eval usually
    )

    print("Starting testing...")
    trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
