"""
Visualize sample images from each class for MyStone, Jonathan El-Beze, and Michel Daudon datasets
"""
import matplotlib.pyplot as plt
from PIL import Image
import os
import numpy as np
import random

def get_random_image(directory, extension):
    """Get a random image file from a directory"""
    files = [f for f in os.listdir(directory) if f.endswith(extension)]
    if files:
        random_file = random.choice(files)
        return os.path.join(directory, random_file)
    return None

def create_mystone_figure():
    """Create figure showing all MyStone classes"""
    base_dir = r'data\MyStone (w256, 1k)\train'
    
    classes = [
        ('1_Subtype_COM_WW_Ia', 'COM_WW_Ia'),
        ('2_Subtype_COD_WD_IIa', 'COD_WD_IIa'),
        ('4_Subtype_HAP', 'HAP'),
        ('5_Subtype_STR_IVc', 'STR_IVc'),
        ('6_Subtype_BRU_IVd', 'BRU_IVd'),
        ('7_Subtype_UA_IIIa', 'UA_IIIa')
    ]
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    # fig.suptitle('MyStone Dataset', fontsize=24)
    
    for idx, (class_dir, label) in enumerate(classes):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        img_path = get_random_image(os.path.join(base_dir, class_dir), '.jpg')
        if img_path:
            img = Image.open(img_path)
            ax.imshow(img)
            ax.set_title(label, fontsize=28)
            ax.set_xticks([])
            ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('mystone_samples.png', dpi=300)
    print("[OK] MyStone figure saved as 'mystone_samples.png'")
    plt.close()

def create_jonathan_figure():
    """Create figure showing all Jonathan El-Beze classes across variants"""
    base_dir = r'data\Jonathan El-Beze (w256 1k v1)'
    
    variants = ['SEC', 'SUR']
    classes = [
        ('Subtype_Ia', 'Ia'),
        ('Subtype_IIa', 'IIa'),
        ('Subtype_IIIa', 'IIIa'),
        ('Subtype_IVc', 'IVc'),
        ('Subtype_IVd', 'IVd'),
        ('Subtype_Va', 'Va')
    ]
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 9))
    # fig.suptitle('Jonathan El-Beze Dataset', 
    #              fontsize=22)
    
    for row, variant in enumerate(variants):
        for col, (class_suffix, label) in enumerate(classes):
            ax = axes[row, col]
            
            class_dir = f'{variant}-{class_suffix}'
            img_dir = os.path.join(base_dir, variant, 'train', class_dir)
            
            img_path = get_random_image(img_dir, '.png')
            if img_path:
                img = Image.open(img_path)
                ax.imshow(img)
                if row == 0:
                    ax.set_title(label, fontsize=28)
                # if row == 2:  # Bottom row
                #     ax.set_xlabel(f'Class {col+1}', fontsize=8)
                if col == 0:  # Left column
                    ax.set_ylabel(variant, fontsize=28, rotation=90)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=12)
                if row == 0:
                    ax.set_title(label, fontsize=18)
                ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('jonathan_samples.png', dpi=300)
    print("[OK] Jonathan El-Beze figure saved as 'jonathan_samples.png'")
    plt.close()

def create_daudon_figure():
    """Create figure showing all Michel Daudon classes across variants"""
    base_dir = r'data\Michel Daudon (w256 1k v1)'
    
    variants = ['SEC', 'SUR']
    classes = [
        ('Subtype_Ia', 'Ia'),
        ('Subtype_IVa', 'IVa'),
        ('Subtype_IVa2', 'IVa2'),
        ('Subtype_IVc', 'IVc'),
        ('Subtype_IVd', 'IVd'),
        ('Subtype_Va', 'Va')
    ]
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 9))
    # fig.suptitle('Michel Daudon Dataset', 
    #              fontsize=24)
    
    for row, variant in enumerate(variants):
        for col, (class_suffix, label) in enumerate(classes):
            ax = axes[row, col]
            
            class_dir = f'{variant}-{class_suffix}'
            img_dir = os.path.join(base_dir, variant, 'train', class_dir)
            
            img_path = get_random_image(img_dir, '.png')
            if img_path:
                img = Image.open(img_path)
                ax.imshow(img)
                if row == 0:
                    ax.set_title(label, fontsize=28)
                # if row == 2:  # Bottom row
                #     ax.set_xlabel(f'Class {col+1}', fontsize=8)
                if col == 0:  # Left column
                    ax.set_ylabel(variant, fontsize=28, rotation=90)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.text(0.5, 0.5, 'Image not found', ha='center', va='center', fontsize=12)
                if row == 0:
                    ax.set_title(label, fontsize=18)
                ax.axis('off')
    
    
    plt.tight_layout()
    plt.savefig('daudon_samples.png', dpi=300)
    print("[OK] Michel Daudon figure saved as 'daudon_samples.png'")
    plt.close()

def main():
    """Generate all three figures"""
    # Set random seed for reproducibility
    random.seed(42)
    
    print("Generating dataset visualization figures...\n")
    
    print("1. Creating MyStone dataset figure...")
    create_mystone_figure()
    
    print("2. Creating Jonathan El-Beze dataset figure...")
    create_jonathan_figure()
    
    print("3. Creating Michel Daudon dataset figure...")
    create_daudon_figure()
    
    print("\n" + "="*60)
    print("All figures generated successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - mystone_samples.png")
    print("  - jonathan_samples.png")
    print("  - daudon_samples.png")

if __name__ == "__main__":
    main()

