<!-- TEMPLATE INSTRUCTIONS: DELETE THIS SECTION BEFORE RELEASE -->
# How to use this template

1.  **Rename**: Rename this folder to your project name.
2.  **Environment**: 
    - Rename `environment.yml` name if needed (default is `torch`).
    - Run `conda env create -f environment.yml`.
3.  **Implement**:
    - Add your model in `models/your_model.py`. It must inherit `pl.LightningModule`.
    - Add your data module in `data/{task}/{dataset}.py` (e.g., `data/classification/mnist.py`). It must inherit `pl.LightningDataModule`.
4.  **Run**:
    - `python train.py model.name=YourModel data.name=MNISTDataModule`
    - Or update `configs/config.yaml` defaults.
5.  **Clean**: Delete this section and update the title below.

---

# Project Name

[Short description of the project]

## Installation

```bash
conda env create -f environment.yml
conda activate torch
```

## Usage

To train the model:

```bash
python train.py
```

### Configuration

You can override parameters from the command line:

```bash
python train.py model.name=MyModel data.name=CIFAR10
```

`data.name` is used for the log directory name, while `data.class_name` is used to import the class.

### Multirun

You can run hyperparameter sweeps using the `-m` or `--multirun` flag:

```bash
python train.py -m max_epochs=5,10 seed=42,43
```

This creates a folder structure organized by the sweep timestamp, then data/model, and finally the job number:

```
logs/
└── multirun/
    └── 2025-12-20_10-00-00/
        ├── TemplateData/TemplateModel/0/
        ├── TemplateData/TemplateModel/1/
        ├── TemplateData/TemplateModel/2/
        └── TemplateData/TemplateModel/3/
```

### Resuming Training

You can resume training from a previously interrupted run using the `resume` parameter. This will automatically:
1.  Load the `last.ckpt` from the checkpoints folder.
2.  Reconnect to the previous WandB run ID.
3.  Continue logging in the same TensorBoard directory.

#### Single Run
Point `resume` to the specific run directory:
```bash
python train.py resume=logs/TemplateData/TemplateModel/2025-12-20_10-00-00
```

#### Multirun
Point `resume` to the root multirun directory. **Note:** You must repeat the original parameters to ensure Hydra reconstructs the job list in the same order.
```bash
python train.py -m max_epochs=10 seed=1,2,3 resume=logs/multirun/2025-12-20_10-00-00
```

## Logging

This project uses both **Weights & Biases (WandB)** and **TensorBoard** for logging.

### Weights & Biases

WandB is enabled by default. The logger is configured as follows:
- **Project**: The name of the current directory (e.g., `ml-template`).
- **Run Name**: `{data.name}/{model.name}`.

**Configuration:**
You can change the WandB mode using the `wandb.mode` parameter:

```bash
# Default (Online)
python train.py

# Offline (save logs locally, sync later)
python train.py wandb.mode=offline

# Disabled (no WandB logging)
python train.py wandb.mode=disabled
```

### TensorBoard

TensorBoard logs are saved locally in the `logs/` directory. Even if WandB is disabled, TensorBoard logging remains active.

To view logs:
```bash
tensorboard --logdir logs/
```

## Evaluation

To evaluate a trained model, provide the path to the checkpoint:

```bash
python eval.py logs/DataModule/Model/.../checkpoints/last.ckpt
```

This script automatically finds the corresponding configuration file in the `.hydra` directory relative to the checkpoint.

## Project Structure

- `configs/`: Hydra configurations.
- `data/`: Data modules organized by task/type (e.g., `data/classification/mnist.py`).
- `models/`: LightningModules.
- `models/networks/`: Neural network architectures (nn.Module).
- `logs/`: TensorBoard logs and checkpoints.
