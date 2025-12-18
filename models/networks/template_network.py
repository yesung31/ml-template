from torch import nn


class TemplateNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError(
            "TemplateNetwork is a placeholder. "
            "Implement your network architecture here."
        )

    def forward(self, x):
        pass
