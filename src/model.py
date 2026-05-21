"""EfficientNet-B0 backbone with a 4-class classifier head."""

import torch.nn as nn
from torchvision import models
from torchvision.models import EfficientNet_B0_Weights


def create_model(num_classes: int = 4) -> models.EfficientNet:
    """
    Build EfficientNet-B0 with ImageNet-pretrained weights and a custom classifier.

    Args:
        num_classes: Number of output classes (default 4 for brain tumor types).

    Returns:
        Model ready for training or inference.
    """
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model
