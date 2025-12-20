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
