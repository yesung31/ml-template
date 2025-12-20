import os
import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

import data
import models
from utils.helpers import instantiate


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    print(f"Training with config:\n{OmegaConf.to_yaml(cfg)}")

    pl.seed_everything(cfg.seed)

    # Dynamic loading of Model and DataModule
    ModelClass = getattr(models, cfg.model.name)
    DataClass = getattr(data, cfg.data.name)

    print(f"Model: {ModelClass.__name__}")
    print(f"DataModule: {DataClass.__name__}")

    # Instantiate
    model = instantiate(ModelClass, cfg.model)
    dm = instantiate(DataClass, cfg.data)

    # Logger
    log_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    logger = TensorBoardLogger(save_dir=log_dir, name="", version="")

    # Checkpointing
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(log_dir, "checkpoints"),
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=10,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()
