import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights


def get_model(num_classes=10, num_channels=3, use_pretrained=True, model_name="resnet18", 
              adapt_for_small_images=True):
    """
    Get a ResNet model and modify it for the specified dataset.
    
    Args:
        num_classes: Number of output classes
        num_channels: Number of input channels (1 for grayscale, 3 for RGB)
        use_pretrained: Whether to use ImageNet pretrained weights
        model_name: Model architecture ("resnet18" or "resnet50")
        adapt_for_small_images: If True, replace conv1 (7x7) and maxpool for small images (<64x64)
    
    Returns:
        Modified ResNet model
    """
    # Load model with or without pretrained weights
    if model_name == "resnet18":
        if use_pretrained:
            model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=None)
    elif model_name == "resnet50":
        if use_pretrained:
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unknown model: {model_name}. Choose from: resnet18, resnet50")
    
    # Adapt for small images (MNIST 28x28, CIFAR 32x32)
    # For images >= 64 (like kidney stones 256x256), use standard architecture
    if adapt_for_small_images:
        # Replace 7x7 conv + stride 2 with 3x3 conv + stride 1
        # This prevents too much downsampling early on
        model.conv1 = nn.Conv2d(
            num_channels, 64, 
            kernel_size=3, stride=1, padding=1, bias=False
        )
        # Remove maxpool (another 2x downsampling) - replace with identity
        model.maxpool = nn.Identity()
    else:
        # For large images (224x224), only adapt channel count if needed
        if num_channels != 3:
            model.conv1 = nn.Conv2d(
                num_channels, 64, 
                kernel_size=7, stride=2, padding=3, bias=False
            )
    
    # Modify the final fully connected layer for the specified number of classes
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    return model
