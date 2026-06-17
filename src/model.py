"""ResNet50-based model definition for plant disease classification."""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

NUM_CLASSES = 38


def build_model(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    """Build a ResNet50 with its final layer replaced for the plant disease classes."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_model(weights_path, num_classes: int = NUM_CLASSES, device=None) -> nn.Module:
    """Build the model architecture and load trained weights from disk."""
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"No trained weights found at {weights_path}. "
            "Train a model and save it there, or place resnet50_plantdisease.pth in the models/ directory."
        )

    device = device or torch.device("cpu")
    model = build_model(num_classes=num_classes, pretrained=False)
    state_dict = torch.load(weights_path, map_location=device)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
