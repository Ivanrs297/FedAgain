"""
Visualize sample images from CIFAR-10 and MNIST datasets
"""
import os
# Fix OpenMP library conflict warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np
import random
import torch

def create_cifar10_figure():
    """Create figure showing all CIFAR-10 classes"""
    # Load CIFAR-10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # CIFAR-10 class names
    classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    
    # Get one random image per class
    for class_idx, class_name in enumerate(classes):
        row = class_idx // 5
        col = class_idx % 5
        ax = axes[row, col]
        
        # Find all images of this class
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx]
        
        # Select random image
        random_idx = random.choice(class_indices)
        img, label = dataset[random_idx]
        
        # Convert tensor to numpy for display (C, H, W) -> (H, W, C)
        img_np = img.permute(1, 2, 0).numpy()
        
        ax.imshow(img_np)
        ax.set_title(class_name, fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('cifar10_samples.png', dpi=300, bbox_inches='tight')
    print("[OK] CIFAR-10 figure saved as 'cifar10_samples.png'")
    plt.close()

def create_mnist_figure():
    """Create figure showing all MNIST classes"""
    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )
    
    # MNIST class names (digits 0-9)
    classes = [str(i) for i in range(10)]
    
    # Create figure
    fig, axes = plt.subplots(2, 5, figsize=(15, 4))
    
    # Get one random image per class
    for class_idx, class_name in enumerate(classes):
        row = class_idx // 5
        col = class_idx % 5
        ax = axes[row, col]
        
        # Find all images of this class
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx]
        
        # Select random image
        random_idx = random.choice(class_indices)
        img, label = dataset[random_idx]
        
        # Convert tensor to numpy for display
        img_np = img.squeeze().numpy()
        
        ax.imshow(img_np, cmap='gray')
        ax.set_title(f'Digit {class_name}', fontsize=16)
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('mnist_samples.png', dpi=300, bbox_inches='tight')
    print("[OK] MNIST figure saved as 'mnist_samples.png'")
    plt.close()

def create_combined_figure():
    """Create a single figure showing both CIFAR-10 and MNIST"""
    # Load datasets
    transform_cifar = transforms.Compose([transforms.ToTensor()])
    transform_mnist = transforms.Compose([transforms.ToTensor()])
    
    cifar_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_cifar
    )
    
    mnist_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform_mnist
    )
    
    # Class names
    cifar_classes = [
        'airplane', 'automobile', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    mnist_classes = [str(i) for i in range(10)]
    
    # Create figure with two rows
    fig = plt.figure(figsize=(20, 8))
    
    # CIFAR-10 row (top)
    for class_idx, class_name in enumerate(cifar_classes):
        ax = plt.subplot(2, 10, class_idx + 1)
        
        # Find random image of this class
        class_indices = [i for i, (_, label) in enumerate(cifar_dataset) if label == class_idx]
        random_idx = random.choice(class_indices)
        img, _ = cifar_dataset[random_idx]
        
        # Display
        img_np = img.permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        ax.set_title(class_name, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add label on left
        if class_idx == 0:
            ax.set_ylabel('CIFAR-10', fontsize=16, rotation=90, labelpad=10)
    
    # MNIST row (bottom)
    for class_idx, class_name in enumerate(mnist_classes):
        ax = plt.subplot(2, 10, 10 + class_idx + 1)
        
        # Find random image of this class
        class_indices = [i for i, (_, label) in enumerate(mnist_dataset) if label == class_idx]
        random_idx = random.choice(class_indices)
        img, _ = mnist_dataset[random_idx]
        
        # Display
        img_np = img.squeeze().numpy()
        ax.imshow(img_np, cmap='gray')
        ax.set_title(f'Digit {class_name}', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add label on left
        if class_idx == 0:
            ax.set_ylabel('MNIST', fontsize=16, rotation=90, labelpad=10)
    
    plt.tight_layout()
    plt.savefig('cifar10_mnist_combined.png', dpi=300, bbox_inches='tight')
    print("[OK] Combined figure saved as 'cifar10_mnist_combined.png'")
    plt.close()

def main():
    """Generate all figures"""
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    print("Generating CIFAR-10 and MNIST visualization figures...\n")
    
    print("1. Creating CIFAR-10 dataset figure...")
    create_cifar10_figure()
    
    print("2. Creating MNIST dataset figure...")
    create_mnist_figure()
    
    print("3. Creating combined figure...")
    create_combined_figure()
    
    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - cifar10_samples.png")
    print("  - mnist_samples.png")
    print("  - cifar10_mnist_combined.png")
    print("\nEach figure shows one random sample from each class.")

if __name__ == "__main__":
    main()

