import torch
import torch.nn as nn
from torchvision import models


class TreeConvNeXtTiny(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.model = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )

        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def crear_modelo(num_classes: int, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    return TreeConvNeXtTiny(num_classes).to(device)