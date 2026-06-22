from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from hydra_auto_resume import auto_resume
from omegaconf import DictConfig

# Calculate absolute path to configs directory
CONFIG_PATH = str(Path(__file__).parent.parent / "configs")


def run(cfg_name="config", eval=False):
    def decorator(func):
        @hydra.main(config_path=CONFIG_PATH, config_name=cfg_name, version_base="1.3")
        @auto_resume(
            resume_arg_name="resume",
            no_log=eval,
            use_saved_config=eval,
        )
        def wrapper(cfg: DictConfig):
            hydra_cfg = hydra.core.hydra_config.HydraConfig.get()
            output_dir = hydra_cfg.runtime.output_dir

            pl.seed_everything(cfg.seed)
            model = hydra.utils.instantiate(cfg.model)
            dm = hydra.utils.instantiate(cfg.data)

            if torch.cuda.is_available():
                torch.set_float32_matmul_precision("medium")

                if any(isinstance(m, torch.nn.Conv2d) for m in model.modules()):
                    model = model.to(memory_format=torch.channels_last)

            if cfg.get("compile", False):
                print("Compiling model...")
                model = torch.compile(model)

            print(f"Model: {cfg.model._target_}")
            print(f"DataModule: {cfg.data._target_}")

            log_dir = Path(output_dir)

            # Use arguments injected by @auto_resume
            ckpt_path = cfg.get("ckpt_path")
            wandb_id = cfg.get("wandb_id")

            return func(cfg, hydra_cfg, model, dm, log_dir, ckpt_path, wandb_id)

        return wrapper

    if callable(cfg_name):
        # The decorator was used without arguments: @run
        func = cfg_name
        cfg_name = "config"
        return decorator(func)

    return decorator
