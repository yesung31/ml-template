from omegaconf import OmegaConf
from pathlib import Path


def load_config_from_ckpt(ckpt_path):
    """
    Attempts to find .hydra/config.yaml relative to the checkpoint path.
    """
    ckpt_path = Path(ckpt_path).resolve()
    # Checkpoint is usually in logs/.../checkpoints/epoch=X.ckpt
    # Config is usually in logs/.../.hydra/config.yaml

    # Traverse up to find .hydra directory
    current = ckpt_path.parent
    for _ in range(4):  # Limit depth check
        config_path = current / ".hydra" / "config.yaml"
        if config_path.exists():
            return OmegaConf.load(config_path)
        current = current.parent

    raise FileNotFoundError(
        f"Could not find .hydra/config.yaml in parent directories of {ckpt_path}"
    )


def instantiate(Class, cfg):
    kwargs = OmegaConf.to_container(cfg, resolve=True)

    # We pass the relevant sub-configs as kwargs, excluding 'name' which is used for class resolution.
    kwargs.pop("name", None)

    return Class(**kwargs)
