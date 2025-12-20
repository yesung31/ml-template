from omegaconf import OmegaConf
from pathlib import Path
from datetime import datetime


def register_resolvers():
    """
    Registers custom OmegaConf resolvers for dynamic path generation.
    """
    if not OmegaConf.has_resolver("resolve_run_dir"):
        OmegaConf.register_new_resolver(
            "resolve_run_dir",
            lambda resume, data, model: resume
            if resume
            else f"logs/{data}/{model}/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        )
    
    if not OmegaConf.has_resolver("resolve_sweep_dir"):
        OmegaConf.register_new_resolver(
            "resolve_sweep_dir",
            lambda resume: resume
            if resume
            else f"logs/multirun/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        )


def load_config_from_ckpt(ckpt_path):
    """
    Attempts to find .hydra/config.yaml relative to the checkpoint path.
    """
    ckpt_path = Path(ckpt_path).resolve()

    # Traverse up to find .hydra directory
    for parent in ckpt_path.parents:
        config_path = parent / ".hydra" / "config.yaml"
        if config_path.exists():
            return OmegaConf.load(config_path)

    raise FileNotFoundError(
        f"Could not find .hydra/config.yaml in parent directories of {ckpt_path}"
    )


def get_resume_info(log_dir):
    """
    Finds the last checkpoint and retrieves the WandB ID from the log directory.
    """
    log_dir = Path(log_dir).resolve()

    # Find checkpoint
    ckpt_path = log_dir / "checkpoints" / "last.ckpt"
    if not ckpt_path.exists():
        print(f"Warning: No 'last.ckpt' found in {ckpt_path.parent}.")
        ckpt_path = None
    else:
        print(f"Resuming from checkpoint: {ckpt_path}")

    # WandB ID handling
    wandb_id = None
    wandb_dir = log_dir / "wandb"
    if wandb_dir.exists():
        # Try finding 'latest-run' symlink first
        latest_run = wandb_dir / "latest-run"
        if latest_run.exists():
            resolved_path = latest_run.resolve()
            wandb_id = resolved_path.name.split("-")[-1]
        else:
            # Fallback: Find the latest 'run-*' or 'offline-run-*' directory
            runs = sorted(
                [
                    d
                    for d in wandb_dir.iterdir()
                    if d.is_dir() and (d.name.startswith("run-") or d.name.startswith("offline-run-"))
                ],
                key=lambda x: x.stat().st_mtime,
            )
            if runs:
                wandb_id = runs[-1].name.split("-")[-1]
        
        if wandb_id:
            print(f"Found previous WandB ID: {wandb_id}")

    return ckpt_path, wandb_id


def instantiate(Class, cfg):
    kwargs = OmegaConf.to_container(cfg, resolve=True)

    # We pass the relevant sub-configs as kwargs, excluding 'name' which is used for class resolution.
    kwargs.pop("name", None)

    return Class(**kwargs)
