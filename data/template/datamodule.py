import warnings

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, TensorDataset


class TemplateDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        warnings.warn(
            "TemplateDataModule is using dummy random data. "
            "Replace this with your actual data loading logic.",
            UserWarning,
        )

    def prepare_data(self):
        # Download data, etc.
        pass

    def setup(self, stage=None):
        # Create dummy data: 100 samples, 32 dimensions
        self.train_dataset = TensorDataset(torch.randn(100, 32), torch.randint(0, 2, (100,)))
        self.val_dataset = TensorDataset(torch.randn(20, 32), torch.randint(0, 2, (20,)))
        self.test_dataset = TensorDataset(torch.randn(20, 32), torch.randint(0, 2, (20,)))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=32)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=32)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=32)
