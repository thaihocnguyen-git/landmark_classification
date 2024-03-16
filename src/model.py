"""Model for image classification, using CNN from scraft
"""
import torch
from torch import nn

def block(in_channels, out_channles, kernel=3, padding='same'):
    conv = nn.Conv2d(in_channels, out_channles, kernel, padding=padding)
    return [
        conv,
        nn.BatchNorm2d(out_channles),
        nn.ReLU(),
        nn.MaxPool2d(2, 2)
    ]

# define the CNN architecture
class MyModel(nn.Module):
    """_summary_

    Args:
        nn (_type_): _description_
    """
    def __init__(self,
                 num_classes: int = 1000,
                 dropout: float = 0.7) -> None:
        super().__init__()
        cnn_features = [
            (3, 32),
            (32, 64),
            (64, 128),
            (128, 256),
            (256, 512),
            (512, 1024)
        ]
        models = []
        for in_feature, out_feature in cnn_features:
            models += block(in_feature, out_feature)

        head = [
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(1024 * 3 * 3, 1024),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, num_classes)
        ]
        models += head
        self.model = nn.Sequential(*models)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = next(dataiter)

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
