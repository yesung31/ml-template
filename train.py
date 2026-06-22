import inspect
import os
import shutil
from pathlib import Path

import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger

from utils.callbacks import MetricAliasCallback, ScientificProgressBar
from utils.results import save_test_results
from utils.run import run


@run
def main(
    cfg: DictConfig,
    hydra_cfg: DictConfig,
    model: LightningModule,
    dm: LightningDataModule,
    log_dir: Path,
    ckpt_path: str,
    wandb_id: str,
):
    choices = hydra_cfg.runtime.choices

    # Backup model source code
    model_file = inspect.getfile(model.__class__)
    target_model_file = log_dir / Path(model_file).name
    if not target_model_file.exists():
        shutil.copy(model_file, log_dir / Path(model_file).name)

    # Loggers
    tb_logger = TensorBoardLogger(save_dir=log_dir, name="", version="")
    
    # name is constructed from data and model choices
    run_name = f"{choices.get('data', 'default')}/{choices.get('model', 'default')}"
    
    wandb_logger = WandbLogger(
        save_dir=log_dir,
        project=Path.cwd().name,
        name=run_name,
        mode=cfg.wandb,
        log_model="all",
        id=wandb_id,
        resume="allow",
    )

    # Callbacks
    # metric_alias = MetricAliasCallback({"val/mcc": "val_mcc"})
    checkpoint_best = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename="best-{epoch}",
        monitor="val_loss", # modify this based on your metric
        mode="min",
        save_top_k=cfg.save_top_k or (1 if cfg.max_epochs < 64 else 3),
    )

    checkpoint_last = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename="last",
    )

    checkpoint_periodic = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        filename="periodic-{epoch}",
        every_n_epochs=cfg.save_every_n_epochs,
        save_top_k=-1,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision="bf16-mixed",
        logger=[tb_logger, wandb_logger],
        callbacks=[
            # metric_alias,
            checkpoint_best,
            checkpoint_last,
            checkpoint_periodic,
            ScientificProgressBar(),
        ],
        log_every_n_steps=10,
        default_root_dir=log_dir,
    )

    trainer.fit(model, dm, ckpt_path=ckpt_path)

    if cfg.test:
        print("Starting testing...")
        results = trainer.test(model, datamodule=dm, ckpt_path="last")

        save_test_results(results, dm, log_dir)

    wandb.finish()


if __name__ == "__main__":
    main()
