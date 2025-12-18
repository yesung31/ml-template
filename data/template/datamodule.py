from pytorch_lightning import LightningDataModule


class TemplateDataModule(LightningDataModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        raise NotImplementedError(
            "TemplateDataModule is a placeholder. "
            "Please implement your own data module in a new file inside the 'data' directory "
            "and update the config or command line arguments to use it."
        )

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
