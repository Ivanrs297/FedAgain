"""
Visualize sample images from MyStone dataset
Shows complete stone fragment images (MIX image set type) and patches extracted from them.
The six subtypes are: Ia, IIa, HAP, IVc, IVd and IIIa
"""
import os
# Fix OpenMP library conflict warning
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
from pathlib import Path

def get_mystone_samples():
    """
    Get random samples from MyStone dataset (MIX image set type)
    Returns dict with class names and sample image paths
    """
    base_path = Path("data/MyStone/MyStone (w256, 1k)/train")
    
    # Get all class folders (sorted for consistency)
    class_folders = sorted([f for f in base_path.iterdir() if f.is_dir()])
    
    samples = []
    
    for class_folder in class_folders:
        # Extract subtype name from folder name
        folder_name = class_folder.name
        
        # Map folder names to subtype names (Ia, IIa, HAP, IVc, IVd, IIIa)
        class_name_map = {
            "1_Subtype_COM_WW_Ia": "Ia",
            "2_Subtype_COD_WD_IIa": "IIa", 
            "4_Subtype_HAP": "HAP",
            "5_Subtype_STR_IVc": "IVc",
            "6_Subtype_BRU_IVd": "IVd",
            "7_Subtype_UA_IIIa": "IIIa"
        }
        
        class_name = class_name_map.get(folder_name, folder_name.split('_')[-1])
        
        # Get paths to images and patches folders
        images_folder = class_folder / "images"
        patch_folder = class_folder / "patch"
        
        # Get random full image (MIX image set type)
        full_images = [f for f in images_folder.glob("*.jpg") if f.is_file()]
        random_full = random.choice(full_images) if full_images else None
        
        # Get random patch image extracted from full images
        patch_images = [f for f in patch_folder.glob("*.jpg") if f.is_file()]
        random_patch = random.choice(patch_images) if patch_images else None
        
        samples.append({
            'class_name': class_name,
            'full_image': random_full,
            'patch_image': random_patch
        })
    
    return samples

def create_mystone_figure():
    """
    Create figure showing MyStone dataset samples
    Layout: 2 rows x 6 columns
    - Row 1: Complete stone fragment images (MIX image set type)
    - Row 2: Patches extracted from the images (small fragment parts)
    """
    # Set random seed for reproducibility
    random.seed(42)
    
    # Get samples
    samples = get_mystone_samples()
    
    if len(samples) != 6:
        print(f"Warning: Expected 6 classes, found {len(samples)}")
    
    # Create figure with 2 rows and 6 columns
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    
    for col_idx, sample in enumerate(samples):
        class_name = sample['class_name']
        full_img_path = sample['full_image']
        patch_img_path = sample['patch_image']
        
        # Row 0: Complete stone fragment images (MIX image set type)
        ax_full = axes[0, col_idx]
        if full_img_path and full_img_path.exists():
            img = Image.open(full_img_path)
            ax_full.imshow(img)
        ax_full.set_xticks([])
        ax_full.set_yticks([])
        
        # Row 1: Patches extracted from images
        ax_patch = axes[1, col_idx]
        if patch_img_path and patch_img_path.exists():
            img = Image.open(patch_img_path)
            ax_patch.imshow(img)
        ax_patch.set_xticks([])
        ax_patch.set_yticks([])
        
        # Add X-axis title at the bottom (on the last row)
        ax_patch.set_xlabel(class_name, fontsize=36)
    
    # Add vertical Y-axis labels on the left
    # axes[0, 0].set_ylabel('MIX', fontsize=20, rotation=90, labelpad=10, verticalalignment='center')
    # axes[1, 0].set_ylabel('Patches', fontsize=20, rotation=90, labelpad=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('mystone_samples.png', dpi=300, bbox_inches='tight')
    print("[OK] MyStone figure saved as 'mystone_samples.png'")
    plt.close()

def main():
    """Generate MyStone visualization figure"""
    print("Generating MyStone dataset visualization...\n")
    
    create_mystone_figure()
    
    print("\n" + "="*60)
    print("Figure generated successfully!")
    print("="*60)
    print("\nGenerated file:")
    print("  - mystone_samples.png")
    print("\nFigure shows:")
    print("  - 6 subtypes (columns): Ia, IIa, HAP, IVc, IVd, IIIa")
    print("  - 2 rows:")
    print("    - Row 1: Complete stone fragment images (MIX image set type)")
    print("    - Row 2: Patches extracted from the images (small fragment parts)")
    print("  - Uniform color background")

if __name__ == "__main__":
    main()
