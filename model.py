import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def get_model(num_classes=10, num_channels=3, use_pretrained=True):
    """
    Get a ResNet18 model and modify it for the specified dataset.
    
    Args:
        num_classes: Number of output classes
        num_channels: Number of input channels (1 for grayscale, 3 for RGB)
        use_pretrained: Whether to use ImageNet pretrained weights
    
    Returns:
        Modified ResNet18 model
    """
    # Load ResNet18 with or without pretrained weights
    if use_pretrained:
        model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    
    # Modify the first convolutional layer if not RGB (3 channels)
    if num_channels != 3:
        # Original: Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Modify the final fully connected layer for the specified number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model

