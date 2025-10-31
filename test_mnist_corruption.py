"""
Quick test to verify MNIST corruption works with the resize fix.
"""
import torch
from torchvision import datasets, transforms
from dataset import CorruptedDataset
import numpy as np

print("="*60)
print("Testing MNIST corruption with grayscale + resize fix")
print("="*60)

# Load MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
])

print("\n1. Loading MNIST dataset...")
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
print(f"   ✓ Loaded {len(mnist)} training images")

# Get a sample to check shape
sample_img, sample_label = mnist[0]
print(f"   ✓ Original MNIST shape: {sample_img.shape} (grayscale, 28x28)")

print("\n2. Creating corrupted dataset (severity=3)...")
corrupted_mnist = CorruptedDataset(mnist, severity=3, seed=42)
print(f"   ✓ Created corrupted dataset")

# Test first few images
print("\n3. Testing first 10 images:")
for i in range(10):
    img, label = corrupted_mnist[i]
    print(f"   Image {i:2d}: shape={img.shape}, label={label}, " +
          f"dtype={img.dtype}, min={img.min():.3f}, max={img.max():.3f}")
    
    # Verify shape
    assert img.shape == (1, 28, 28), f"Wrong shape: {img.shape}"
    assert img.dtype == torch.float32, f"Wrong dtype: {img.dtype}"
    assert 0 <= img.min() <= 1, f"Values out of range: min={img.min()}"
    assert 0 <= img.max() <= 1, f"Values out of range: max={img.max()}"

print("\n" + "="*60)
print("✅ SUCCESS! MNIST corruption working correctly.")
print("="*60)
print("Details:")
print("  - Grayscale (1 channel) preserved")
print("  - Size 28x28 preserved")
print("  - Images temporarily converted to RGB 32x32 for corruption")
print("  - Then converted back to grayscale 28x28")
print("  - Values normalized to [0, 1] range")
print("="*60)

