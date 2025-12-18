from pytorch_lightning import LightningModule


class TemplateModel(LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        raise NotImplementedError(
            "TemplateModel is a placeholder. "
            "Please implement your own model in a new file inside the 'models' directory "
            "and update the config or command line arguments to use it."
        )

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass
