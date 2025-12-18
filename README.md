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
    - `python train.py model.class_name=YourModel data.class_name=classification.mnist.MNISTDataModule`
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
python train.py model.name=MyModel data.name=CIFAR10 data.class_name=classification.cifar.CIFAR10
```

`data.name` is used for the log directory name, while `data.class_name` is used to import the class.

## Evaluation

To evaluate a trained model, provide the path to the checkpoint:

```bash
python eval.py logs/template.datamodule.TemplateDataModule/TemplateModel/.../checkpoints/last.ckpt
```

This script automatically finds the corresponding configuration file in the `.hydra` directory relative to the checkpoint.

## Project Structure

- `configs/`: Hydra configurations.
- `data/`: Data modules organized by task/type (e.g., `data/classification/mnist.py`).
- `models/`: LightningModules.
- `models/networks/`: Neural network architectures (nn.Module).
- `logs/`: TensorBoard logs and checkpoints.
