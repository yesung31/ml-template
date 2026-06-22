<!-- TEMPLATE INSTRUCTIONS: DELETE THIS SECTION BEFORE RELEASE -->
# How to use this template

1.  **Rename**: Rename this folder to your project name.
2.  **Environment**: 
    - This project uses `uv` for lightning-fast dependency management.
    - Run `uv sync` to install all dependencies from `pyproject.toml`.
3.  **Implement**:
    - Add your model in `models/your_model.py`. It must inherit `pl.LightningModule`.
    - Add your data module in `data/{task}/{dataset}.py` (e.g., `data/classification/mnist.py`). It must inherit `pl.LightningDataModule`.
    - Configure them in `configs/model` and `configs/data` by specifying the `_target_` key for Hydra instantiation.
4.  **Run**:
    - `uv run python train.py model=your_model data=mnist`
    - Or update `configs/config.yaml` defaults.
5.  **Clean**: Delete this section and update the title below.

---

# Project Name

[Short description of the project]

## Features
- **Boilerplate-free**: Powered by the custom `@run` decorator, PyTorch Lightning training loops, logging, automatic resuming, and model compilation are automatically handled.
- **Hydra Configuration**: Full integration with Hydra for dynamic instantiation of models and data modules via the `_target_` key.
- **Auto-resuming**: Robust, seamless checkpoint resuming with `hydra-auto-resume`.
- **HPC Ready**: Pre-configured with `hydra-submitit-launcher` for easy Slurm cluster scaling.

## Installation

We use `uv` as the package manager for speed and determinism.

```bash
# Sync and install the virtual environment
uv sync

# (Optional) activate the environment manually
source .venv/bin/activate
```

## Usage

To train the model:

```bash
uv run python train.py
```

### Configuration

You can override parameters from the command line:

```bash
uv run python train.py model=TemplateModel data=TemplateDataModule
```

Models and data modules are dynamically instantiated by Hydra using their `_target_` paths defined in `configs/model/` and `configs/data/`.

### Advanced Launching (Slurm & Ablations)

You can launch hyperparameter sweeps natively.

```bash
uv run python train.py -m max_epochs=5,10 seed=42,43
```

To run on a Slurm cluster, use the provided submitit launcher:
```bash
uv run python train.py -m hydra/launcher=slurm max_epochs=50
```

### Resuming Training

We use `hydra-auto-resume`. You can resume training from a previously interrupted run by simply pointing the `resume` parameter to the run directory. This will automatically load the checkpoint, reconnect WandB, and restore the previous Hydra configurations!

```bash
uv run python train.py resume=logs/TemplateDataModule/TemplateModel/2026-06-22_10-00-00
```

## Logging

This project uses both **Weights & Biases (WandB)** and **TensorBoard** for logging.

### Weights & Biases

WandB is enabled by default. The logger is configured automatically by the `@run` decorator:
- **Project**: The name of the current directory.
- **Run Name**: Automatically inferred from data and model choices.

**Configuration:**
You can change the WandB mode using the `wandb` parameter:

```bash
# Default (Online)
uv run python train.py

# Offline (save logs locally, sync later)
uv run python train.py wandb=offline

# Disabled (no WandB logging)
uv run python train.py wandb=disabled
```

## Evaluation

The evaluation script relies on the exact same `@run` environment, ensuring consistent configurations. 
To evaluate a trained model, provide the checkpoint path:

```bash
uv run python eval.py ckpt_path=logs/TemplateDataModule/TemplateModel/.../checkpoints/last.ckpt
```

This will run testing, calculate metrics, and output a structured `test_results.json` directly into your log directory.

## Project Structure

- `configs/`: Hydra configurations organized intuitively.
- `data/`: Data modules (LightningDataModule).
- `models/`: LightningModules.
- `utils/`: Core utilities like the `@run` decorator, advanced callbacks, and evaluation logic.
- `logs/`: Checkpoints, source-code backups, TensorBoard logs, and test results.
