from pathlib import Path

import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule

from utils.callbacks import ScientificProgressBar
from utils.results import save_test_results
from utils.run import run


@run(eval=True)
def main(
    cfg: DictConfig,
    hydra_cfg: DictConfig,
    model: LightningModule,
    dm: LightningDataModule,
    log_dir: Path,
    ckpt_path: str,
    wandb_id: str,
):
    trainer = pl.Trainer(
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        precision="bf16-mixed",
        callbacks=[ScientificProgressBar(refresh_rate=1, metric_update_interval=10)],
        default_root_dir=log_dir,
        logger=False,
        enable_checkpointing=False,
    )

    print(f"Starting testing using checkpoint: {ckpt_path}")
    results = trainer.test(model, dm, ckpt_path=ckpt_path)

    # Save formatted results
    save_test_results(results, dm, log_dir)


if __name__ == "__main__":
    main()
