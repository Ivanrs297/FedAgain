import torch
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from typing import Tuple, List, Dict
import os
import numpy as np
from PIL import Image
import warnings

# Suppress deprecation warnings from imagecorruptions package
warnings.filterwarnings('ignore', category=FutureWarning, module='imagecorruptions')
warnings.filterwarnings('ignore', category=UserWarning, message='.*non-writable tensors.*')

try:
    from imagecorruptions import corrupt, get_corruption_names
    IMAGECORRUPTIONS_AVAILABLE = True
except ImportError:
    IMAGECORRUPTIONS_AVAILABLE = False
    print("[WARNING] imagecorruptions package not found. Install with: pip install imagecorruptions")


# ---------------------------------------------------------------------------
# Data version (second MedIA revision)
#   v1 : the released train/test folders as loaded until 2026-10 (MyStone read
#        through ImageFolder, which also picked up its full frames and binary
#        segmentation masks; private splits not specimen-disjoint).
#   v2 : specimen-disjoint splits built by scripts/build_specimen_splits.py
#        (data/splits/*_v2.csv), MyStone patches only, plus KIDNEY_MULTISITE.
# Resolution order: FEDAGAIN_DATA_VERSION env var > config.DATA_VERSION > "v1".
# ---------------------------------------------------------------------------
KIDNEY_V2 = {
    "JONATHAN_MIX": "JONATHAN_MIX", "JONATHAN_SEC": "JONATHAN_SEC",
    "JONATHAN_SUR": "JONATHAN_SUR", "DAUDON_MIX": "DAUDON_MIX",
    "DAUDON_SEC": "DAUDON_SEC", "DAUDON_SUR": "DAUDON_SUR", "MYSTONE": "MYSTONE",
}
# Natural three-site federation: each collection is one acquisition site.
MULTISITE_SOURCES = [("A", "DAUDON_MIX"), ("B", "JONATHAN_MIX"), ("C", "MYSTONE")]


def data_version():
    v = os.environ.get("FEDAGAIN_DATA_VERSION")
    if not v:
        try:
            import config as _cfg
            v = getattr(_cfg, "DATA_VERSION", "v1")
        except ImportError:
            v = "v1"
    return v


def subtype_of(class_name):
    """Harmonized subtype label: 'MIX-Subtype_IVa2' -> 'IVa2',
    '1_Subtype_COM_WW_Ia' -> 'Ia', '4_Subtype_HAP' -> 'HAP'."""
    return class_name.split("_")[-1]


class ListImageDataset(Dataset):
    """Image dataset from an explicit (path, label) list, ImageFolder-compatible
    (``classes``, ``targets``), with optional per-sample site and specimen group."""

    def __init__(self, paths, targets, classes, transform=None, sites=None, groups=None):
        self.paths = list(paths)
        self.targets = [int(t) for t in targets]
        self.classes = list(classes)
        self.transform = transform
        self.sites = np.array(sites) if sites is not None else None
        self.groups = np.array(groups) if groups is not None else None

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]


def _read_split_csv(name, root):
    import csv
    path = os.path.join(root, "splits", f"{name}_v2.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found: run `python scripts/build_specimen_splits.py` first")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _load_kidney_v2(dataset_name, transform, root):
    """Specimen-disjoint v2 train/test sets (class indices follow the sorted
    folder names, exactly as ImageFolder assigned them in v1)."""
    rows = _read_split_csv(KIDNEY_V2[dataset_name], root)
    classes = sorted({r["class_name"] for r in rows})
    cidx = {c: i for i, c in enumerate(classes)}
    out = []
    for split in ("train", "test"):
        sel = [r for r in rows if r["split"] == split]
        out.append(ListImageDataset(
            [os.path.join(root, r["relpath"]) for r in sel],
            [cidx[r["class_name"]] for r in sel], classes, transform,
            groups=[r["group"] for r in sel]))
    return out[0], out[1]


def _load_multisite_v2(transform, root):
    """KIDNEY_MULTISITE: the three collections as three acquisition sites, on the
    union of their subtypes (9 classes, no merging: each site lacks some)."""
    per_site = []
    for site, name in MULTISITE_SOURCES:
        per_site.append((site, _read_split_csv(KIDNEY_V2[name], root)))
    classes = sorted({subtype_of(r["class_name"]) for _, rows in per_site for r in rows})
    cidx = {c: i for i, c in enumerate(classes)}
    out = []
    for split in ("train", "test"):
        paths, targets, sites, groups = [], [], [], []
        for site, rows in per_site:
            for r in rows:
                if r["split"] != split:
                    continue
                paths.append(os.path.join(root, r["relpath"]))
                targets.append(cidx[subtype_of(r["class_name"])])
                sites.append(site)
                groups.append(f"{site}:{r['group']}")
        out.append(ListImageDataset(paths, targets, classes, transform,
                                    sites=sites, groups=groups))
    return out[0], out[1]


class CorruptedDataset(Dataset):
    """
    Wrapper to apply random image corruptions to a dataset.
    Simulates corrupted/poisoned clients in federated learning.
    
    Each image is corrupted with a random corruption type at specified severity.
    Uses imagecorruptions package (pip install imagecorruptions).
    """
    
    def __init__(self, base_dataset, severity=3, seed=42):
        """
        Args:
            base_dataset: Original clean dataset
            severity: Corruption severity (1-5, where 5 is most severe)
            seed: Random seed for reproducibility
        """
        self.base_dataset = base_dataset
        self.severity = severity
        
        if not IMAGECORRUPTIONS_AVAILABLE:
            raise ImportError("imagecorruptions package required. Install with: pip install imagecorruptions")
        
        # Get available corruption types (19 corruptions)
        self.corruption_names = get_corruption_names('all')
        
        # Randomly assign corruption types to images
        np.random.seed(seed)
        self.corruption_types = np.random.choice(
            self.corruption_names,
            size=len(base_dataset),
            replace=True
        )
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        # Get original image and label
        img, label = self.base_dataset[idx]
        
        # Convert tensor to numpy if needed
        if isinstance(img, torch.Tensor):
            # Tensor to numpy: (C, H, W) -> (H, W, C)
            img_np = img.permute(1, 2, 0).numpy()
            # Denormalize if needed (assume 0-1 range)
            img_np = (img_np * 255).astype(np.uint8)
            # Squeeze single channel dimension for grayscale: (H, W, 1) -> (H, W)
            if img_np.shape[2] == 1:
                img_np = img_np.squeeze(2)
            was_tensor = True
        else:
            # PIL Image to numpy
            img_np = np.array(img)
            was_tensor = False
        
        # Check image size and resize if needed (imagecorruptions requires 32x32 minimum)
        original_size = img_np.shape[:2]  # (H, W)
        is_grayscale = len(img_np.shape) == 2
        needs_resize = original_size[0] < 32 or original_size[1] < 32
        
        if needs_resize:
            # Resize to 32x32 for corruption
            from PIL import Image as PILImage
            if is_grayscale:  # Grayscale
                img_pil = PILImage.fromarray(img_np, mode='L')
            else:  # RGB
                img_pil = PILImage.fromarray(img_np)
            img_pil_resized = img_pil.resize((32, 32), PILImage.BILINEAR)
            img_np_resized = np.array(img_pil_resized)
        else:
            img_np_resized = img_np
        
        # Apply corruption
        # Note: imagecorruptions expects RGB (3-channel) images
        # Convert grayscale to RGB temporarily if needed
        if is_grayscale and len(img_np_resized.shape) == 2:
            # Convert grayscale to RGB by repeating channels
            img_np_resized_rgb = np.stack([img_np_resized] * 3, axis=2)
        else:
            img_np_resized_rgb = img_np_resized
        
        corruption_type = self.corruption_types[idx]
        try:
            corrupted_np_rgb = corrupt(img_np_resized_rgb, corruption_name=corruption_type, severity=self.severity)
            # Convert back to grayscale if original was grayscale
            if is_grayscale:
                # Use weighted average (standard RGB to grayscale conversion)
                corrupted_np = np.dot(corrupted_np_rgb[...,:3], [0.299, 0.587, 0.114]).astype(np.uint8)
            else:
                corrupted_np = corrupted_np_rgb
        except (TypeError, AttributeError) as e:
            # Handle scikit-image compatibility issue or size issues
            # If corruption fails, skip corruption for this image
            print(f"[WARNING] Corruption '{corruption_type}' failed: {e}, using clean image")
            corrupted_np = img_np_resized
        
        # Resize back to original size if needed
        if needs_resize:
            from PIL import Image as PILImage
            if len(corrupted_np.shape) == 2:  # Grayscale
                corrupted_pil = PILImage.fromarray(corrupted_np, mode='L')
            else:  # RGB
                corrupted_pil = PILImage.fromarray(corrupted_np)
            corrupted_pil_resized = corrupted_pil.resize(original_size[::-1], PILImage.BILINEAR)  # (W, H)
            corrupted_np = np.array(corrupted_pil_resized)
        
        # Convert back to original format
        if was_tensor:
            # Numpy to tensor: (H, W, C) -> (C, H, W)
            # Make a copy to ensure writable array
            corrupted_np = np.array(corrupted_np, copy=True)
            # Handle grayscale images (add channel dimension if needed)
            if len(corrupted_np.shape) == 2:
                corrupted_np = corrupted_np[:, :, np.newaxis]
            corrupted_tensor = torch.from_numpy(corrupted_np).permute(2, 0, 1).float() / 255.0
            return corrupted_tensor, label
        else:
            # Numpy to PIL
            if len(corrupted_np.shape) == 2:  # Grayscale
                corrupted_pil = Image.fromarray(corrupted_np, mode='L')
            else:  # RGB
                corrupted_pil = Image.fromarray(corrupted_np)
            return corrupted_pil, label


class LabelFlipDataset(Dataset):
    """
    Wrapper that applies a label-flipping data-poisoning attack to a client.

    Unlike CorruptedDataset (which degrades the *images*), this is an adversarial
    attack on the *labels*: a fraction of the client's samples have their target
    replaced by a wrong class. This is the classic label-flipping poisoning
    attack, distinct from the on-the-fly image corruption that simulates data
    quality issues. The image itself is returned unchanged.

    The flipped label is deterministic given (idx, seed): a per-sample offset in
    [1, num_classes-1] is added modulo num_classes, so the target is always moved
    to a different, reproducibly-chosen class (a "dirty-label" flip). With
    flip_ratio=1.0 every sample of the poisoned client is mislabeled.
    """

    def __init__(self, base_dataset, num_classes, flip_ratio=1.0, seed=42):
        """
        Args:
            base_dataset: Original clean client dataset
            num_classes: Number of classes in the task (targets in [0, C-1])
            flip_ratio: Fraction of the client's samples to mislabel (0-1)
            seed: Random seed for reproducibility (vary per client)
        """
        if num_classes is None or num_classes < 2:
            raise ValueError("LabelFlipDataset requires num_classes >= 2")
        self.base_dataset = base_dataset
        self.num_classes = int(num_classes)
        n = len(base_dataset)
        rng = np.random.RandomState(seed)
        # Which samples get flipped, and to which (different) class.
        self.flip = rng.random(n) < flip_ratio
        # Offset in [1, C-1] guarantees the new label differs from the original.
        self.offset = rng.randint(1, self.num_classes, size=n)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if self.flip[idx]:
            label = int((int(label) + int(self.offset[idx])) % self.num_classes)
        return img, label


class CIFAR10CDataset(Dataset):
    """
    CIFAR-10-C with Random Corruptions: Realistic robustness training and testing.
    
    Each image (both train and test) has a RANDOM corruption type applied.
    This simulates real-world scenarios where:
    - Training data is noisy/corrupted (realistic conditions)
    - Test data is also corrupted but with different corruption instances
    - Model must learn to be robust from corrupted training data
    
    Both sets use:
    - All 19 corruption types randomly distributed
    - Severity level 3 (medium)
    - One corruption per image
    """
    
    CORRUPTION_TYPES = [
        'gaussian_noise', 'shot_noise', 'impulse_noise', 'speckle_noise',
        'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'gaussian_blur',
        'snow', 'frost', 'fog',
        'brightness', 'contrast', 'elastic_transform', 'pixelate', 
        'jpeg_compression', 'spatter', 'saturate'
    ]
    
    def __init__(self, root, severity=3, transform=None, train=True, seed=42):
        """
        Args:
            root: Path to CIFAR-10-C directory (should contain .npy files)
            severity: Severity level (1-5, where 5 is most severe)
            transform: Optional transform to apply
            train: If True, load training set; if False, load test set
            seed: Random seed for reproducibility (different seeds for train/test)
        """
        self.root = root
        self.severity = severity
        self.transform = transform
        self.train = train
        
        # Use different seeds for train and test to ensure different corruption instances
        self.seed = seed if train else seed + 1000
        
        # Determine which CIFAR-10 split to use
        split_name = "train" if train else "test"
        num_images = 50000 if train else 10000
        
        # Check if CIFAR-10-C exists
        labels_file = os.path.join(root, "labels.npy")
        if not os.path.exists(labels_file):
            raise FileNotFoundError(
                f"CIFAR-10-C not found at: {root}\n"
                f"Please download from: "
                f"https://zenodo.org/record/2535967/files/CIFAR-10-C.tar"
            )
        
        # Load original CIFAR-10 labels for this split
        cifar10 = datasets.CIFAR10(
            root=os.path.dirname(root),
            train=train,
            download=True
        )
        self.targets = cifar10.targets if hasattr(cifar10, 'targets') else cifar10.labels
        
        # Randomly assign corruption types to images
        np.random.seed(self.seed)
        corruption_assignments = np.random.choice(
            len(self.CORRUPTION_TYPES), 
            size=num_images, 
            replace=True
        )
        
        # Load all corruption data (only test set available in CIFAR-10-C)
        # For training, we'll use the test corruptions but with different indices
        print(f"\nLoading CIFAR-10-C with random corruptions (severity {severity}, {split_name} set)...")
        
        corruption_data = {}
        start_idx = (severity - 1) * 10000
        end_idx = severity * 10000
        
        for i, corruption in enumerate(self.CORRUPTION_TYPES):
            corruption_file = os.path.join(root, f"{corruption}.npy")
            
            if not os.path.exists(corruption_file):
                print(f"  [WARNING] Missing {corruption}.npy, skipping...")
                continue
            
            # Load this corruption's data at specified severity
            data = np.load(corruption_file)
            corruption_data[i] = data[start_idx:end_idx]
            
            if (i + 1) % 5 == 0:
                print(f"  Loaded {i+1}/{len(self.CORRUPTION_TYPES)} corruption types...")
        
        # Build dataset: each image gets its assigned corruption
        self.data = []
        self.corruption_names = []
        
        if train:
            # For training: tile the test corruptions to cover 50k images
            # Each test corruption has 10k images, we need 50k
            # We'll use tiling + random selection
            for img_idx, corruption_idx in enumerate(corruption_assignments):
                if corruption_idx in corruption_data:
                    # Map training index to corruption data index (cycle through 10k)
                    data_idx = img_idx % 10000
                    self.data.append(corruption_data[corruption_idx][data_idx])
                    self.corruption_names.append(self.CORRUPTION_TYPES[corruption_idx])
                else:
                    # Fallback
                    fallback_idx = list(corruption_data.keys())[0]
                    data_idx = img_idx % 10000
                    self.data.append(corruption_data[fallback_idx][data_idx])
                    self.corruption_names.append(self.CORRUPTION_TYPES[fallback_idx])
        else:
            # For testing: use the 10k test images directly
            for img_idx, corruption_idx in enumerate(corruption_assignments):
                if corruption_idx in corruption_data:
                    self.data.append(corruption_data[corruption_idx][img_idx])
                    self.corruption_names.append(self.CORRUPTION_TYPES[corruption_idx])
                else:
                    # Fallback
                    fallback_idx = list(corruption_data.keys())[0]
                    self.data.append(corruption_data[fallback_idx][img_idx])
                    self.corruption_names.append(self.CORRUPTION_TYPES[fallback_idx])
        
        self.data = np.array(self.data)
        
        # Print corruption distribution
        unique, counts = np.unique(self.corruption_names, return_counts=True)
        print(f"\n  [OK] Created {split_name} set with {len(self.data)} images")
        print(f"  Corruption distribution (random, severity {severity}):")
        for corruption, count in sorted(zip(unique, counts)):
            print(f"    {corruption:20s}: {count:5d} images ({count/len(self.data)*100:.1f}%)")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]
        
        # Convert to PIL Image
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, target


def get_dataset_info(dataset_name: str) -> Dict:
    """
    Get dataset-specific information.
    
    Args:
        dataset_name: Name of the dataset ("EMNIST", "MNIST", "CIFAR10")
    
    Returns:
        Dictionary with dataset info (num_classes, num_channels, input_size, mean, std)
    """
    dataset_configs = {
        "EMNIST": {
            "num_classes": 62,
            "num_channels": 1,
            "input_size": 28,  # Native size - no upscaling!
            "mean": [0.5],
            "std": [0.5]
        },
        "MNIST": {
            "num_classes": 10,
            "num_channels": 1,
            "input_size": 28,  # Native size - no upscaling!
            "mean": [0.5],
            "std": [0.5]
        },
        "CIFAR10": {
            "num_classes": 10,
            "num_channels": 3,
            "input_size": 32,  # Native size - no upscaling!
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "CIFAR100": {
            "num_classes": 100,
            "num_channels": 3,
            "input_size": 32,  # Native size - no upscaling!
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "CIFAR10C": {
            "num_classes": 10,
            "num_channels": 3,
            "input_size": 32,  # Native size - no upscaling!
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
            "severity": 3  # Fixed at 3 for consistency
        },
        # Kidney Stone Datasets - Jonathan El-Beze
        "JONATHAN_MIX": {
            "num_classes": 6,  # 6 subtypes: Ia, IIa, IIIa, IVc, IVd, Va
            "num_channels": 3,
            "input_size": 256,
            "mean": [0.485, 0.456, 0.406],  # ImageNet normalization
            "std": [0.229, 0.224, 0.225]
        },
        "JONATHAN_SEC": {
            "num_classes": 6,
            "num_channels": 3,
            "input_size": 256,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "JONATHAN_SUR": {
            "num_classes": 6,
            "num_channels": 3,
            "input_size": 256,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        # Kidney Stone Datasets - Michel Daudon
        "DAUDON_MIX": {
            "num_classes": 6,  # 6 subtypes: Ia, IVa, IVa2, IVc, IVd, Va
            "num_channels": 3,
            "input_size": 256,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "DAUDON_SEC": {
            "num_classes": 6,
            "num_channels": 3,
            "input_size": 256,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "DAUDON_SUR": {
            "num_classes": 6,
            "num_channels": 3,
            "input_size": 256,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        # Kidney Stone Dataset - MyStone
        "MYSTONE": {
            "num_classes": 6,  # 6 subtypes
            "num_channels": 3,
            "input_size": 256,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        # v2 only: Daudon + Jonathan + MyStone as three sites, union of subtypes
        "KIDNEY_MULTISITE": {
            "num_classes": 9,  # Ia IIa IIIa IVa IVa2 IVc IVd Va HAP
            "num_channels": 3,
            "input_size": 256,
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }
    }
    
    if dataset_name not in dataset_configs:
        raise ValueError(f"Dataset {dataset_name} not supported. Choose from: {list(dataset_configs.keys())}")
    
    return dataset_configs[dataset_name]


def load_dataset(dataset_name: str, root: str = "./data"):
    """
    Load the specified dataset.
    
    Args:
        dataset_name: Name of the dataset ("EMNIST", "MNIST", "CIFAR10")
        root: Root directory to store the dataset
    
    Returns:
        Tuple of (train_dataset, test_dataset, dataset_info)
    """
    dataset_info = get_dataset_info(dataset_name)
    
    # Define transforms based on dataset
    transform = transforms.Compose([
        transforms.Resize((dataset_info["input_size"], dataset_info["input_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=dataset_info["mean"], std=dataset_info["std"])
    ])

    if dataset_name == "KIDNEY_MULTISITE" or (data_version() == "v2" and dataset_name in KIDNEY_V2):
        if data_version() != "v2":
            raise ValueError("KIDNEY_MULTISITE exists only in data version v2 "
                             "(set FEDAGAIN_DATA_VERSION=v2)")
        if dataset_name == "KIDNEY_MULTISITE":
            train_dataset, test_dataset = _load_multisite_v2(transform, root)
        else:
            train_dataset, test_dataset = _load_kidney_v2(dataset_name, transform, root)
        print(f"  - Data version v2: {len(train_dataset)} train / {len(test_dataset)} test "
              f"samples, specimen-disjoint")
        return train_dataset, test_dataset, dataset_info

    if dataset_name == "EMNIST":
        train_dataset = datasets.EMNIST(
            root=root,
            split="byclass",
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.EMNIST(
            root=root,
            split="byclass",
            train=False,
            download=True,
            transform=transform
        )
    
    elif dataset_name == "MNIST":
        train_dataset = datasets.MNIST(
            root=root,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.MNIST(
            root=root,
            train=False,
            download=True,
            transform=transform
        )
    
    elif dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(
            root=root,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.CIFAR10(
            root=root,
            train=False,
            download=True,
            transform=transform
        )
    
    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(
            root=root,
            train=True,
            download=True,
            transform=transform
        )
        test_dataset = datasets.CIFAR100(
            root=root,
            train=False,
            download=True,
            transform=transform
        )
    
    elif dataset_name == "CIFAR10C":
        # CIFAR-10-C: BOTH training and testing use randomly corrupted images
        # Each image gets ONE random corruption type at severity 3
        # Training: 50k images with random corruptions
        # Testing: 10k images with random corruptions (different instances)
        import config
        severity = getattr(config, "CIFAR10C_SEVERITY", 3)
        
        cifar10c_root = os.path.join(root, "CIFAR-10-C")
        
        train_dataset = CIFAR10CDataset(
            root=cifar10c_root,
            severity=severity,
            transform=transform,
            train=True
        )
        test_dataset = CIFAR10CDataset(
            root=cifar10c_root,
            severity=severity,
            transform=transform,
            train=False
        )
    
    # Jonathan El-Beze datasets
    elif dataset_name == "JONATHAN_MIX":
        train_dir = os.path.join(root, "Jonathan El-Beze (w256 1k v1)", "MIX", "train")
        test_dir = os.path.join(root, "Jonathan El-Beze (w256 1k v1)", "MIX", "test")
        train_dataset = ImageFolder(train_dir, transform=transform)
        test_dataset = ImageFolder(test_dir, transform=transform)
    
    elif dataset_name == "JONATHAN_SEC":
        train_dir = os.path.join(root, "Jonathan El-Beze (w256 1k v1)", "SEC", "train")
        test_dir = os.path.join(root, "Jonathan El-Beze (w256 1k v1)", "SEC", "test")
        train_dataset = ImageFolder(train_dir, transform=transform)
        test_dataset = ImageFolder(test_dir, transform=transform)
    
    elif dataset_name == "JONATHAN_SUR":
        train_dir = os.path.join(root, "Jonathan El-Beze (w256 1k v1)", "SUR", "train")
        test_dir = os.path.join(root, "Jonathan El-Beze (w256 1k v1)", "SUR", "test")
        train_dataset = ImageFolder(train_dir, transform=transform)
        test_dataset = ImageFolder(test_dir, transform=transform)
    
    # Michel Daudon datasets
    elif dataset_name == "DAUDON_MIX":
        train_dir = os.path.join(root, "Michel Daudon (w256 1k v1)", "MIX", "train")
        test_dir = os.path.join(root, "Michel Daudon (w256 1k v1)", "MIX", "test")
        train_dataset = ImageFolder(train_dir, transform=transform)
        test_dataset = ImageFolder(test_dir, transform=transform)
    
    elif dataset_name == "DAUDON_SEC":
        train_dir = os.path.join(root, "Michel Daudon (w256 1k v1)", "SEC", "train")
        test_dir = os.path.join(root, "Michel Daudon (w256 1k v1)", "SEC", "test")
        train_dataset = ImageFolder(train_dir, transform=transform)
        test_dataset = ImageFolder(test_dir, transform=transform)
    
    elif dataset_name == "DAUDON_SUR":
        train_dir = os.path.join(root, "Michel Daudon (w256 1k v1)", "SUR", "train")
        test_dir = os.path.join(root, "Michel Daudon (w256 1k v1)", "SUR", "test")
        train_dataset = ImageFolder(train_dir, transform=transform)
        test_dataset = ImageFolder(test_dir, transform=transform)
    
    # MyStone dataset
    elif dataset_name == "MYSTONE":
        train_dir = os.path.join(root, "MyStone", "MyStone (w256, 1k)", "train")
        test_dir = os.path.join(root, "MyStone", "MyStone (w256, 1k)", "test")
        train_dataset = ImageFolder(train_dir, transform=transform)
        test_dataset = ImageFolder(test_dir, transform=transform)
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    return train_dataset, test_dataset, dataset_info


def apply_corruptions_to_clients(client_datasets, corrupt_ratio=0.3, severity=3, seed=42,
                                 poison_mode="image", num_classes=None, flip_ratio=1.0):
    """
    Apply corruptions to a subset of clients (simulating corrupted/poisoned clients).

    Args:
        client_datasets: List of client datasets
        corrupt_ratio: Ratio of clients to corrupt (e.g., 0.3 = 30%)
        severity: Corruption severity (1-5), used when poison_mode == "image"
        seed: Random seed controlling which clients are corrupted and their
              per-client corruption instances (varies across experiment seeds)
        poison_mode: "image" (on-the-fly image corruption, the default) or
              "label_flip" (adversarial label-flipping data-poisoning attack)
        num_classes: Number of task classes, required for poison_mode="label_flip"
        flip_ratio: Fraction of a poisoned client's labels to flip (label_flip only)

    Returns:
        List of datasets (some corrupted, some clean)
        List of corrupted client indices

    The corrupted-client SELECTION is identical across poison modes (same seeded
    np.random.choice), so downstream analyses that reconstruct which clients were
    poisoned from the seed remain valid regardless of the attack type.
    """
    import numpy as np

    num_clients = len(client_datasets)
    num_corrupt = int(num_clients * corrupt_ratio)

    # Randomly select clients to corrupt (identical selection for every poison_mode)
    np.random.seed(seed)
    corrupted_indices = np.random.choice(num_clients, num_corrupt, replace=False)

    print(f"\n[Corrupted Clients Simulation]")
    print(f"  Total clients: {num_clients}")
    print(f"  Corrupted clients: {num_corrupt} ({corrupt_ratio*100:.0f}%)")
    print(f"  Corrupted client IDs: {sorted(corrupted_indices.tolist())}")
    print(f"  Poison mode: {poison_mode}"
          + (f" (flip_ratio={flip_ratio})" if poison_mode == "label_flip"
             else f" (severity={severity})"))
    print(f"  Benign clients: {num_clients - num_corrupt} ({(1-corrupt_ratio)*100:.0f}%)")

    # Apply the chosen poisoning to selected clients
    result_datasets = []
    for i, dataset in enumerate(client_datasets):
        if i in corrupted_indices:
            if poison_mode == "label_flip":
                poisoned = LabelFlipDataset(
                    dataset,
                    num_classes=num_classes,
                    flip_ratio=flip_ratio,
                    seed=seed + i  # Different flip pattern per client
                )
            else:
                poisoned = CorruptedDataset(
                    dataset,
                    severity=severity,
                    seed=seed + i  # Different corruption per client
                )
            result_datasets.append(poisoned)
        else:
            # Keep clean
            result_datasets.append(dataset)

    return result_datasets, corrupted_indices.tolist()


def partition_data_iid(dataset, num_clients: int, seed: int = 42):
    """IID partition: randomly split data equally among clients."""
    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * num_clients
    lengths[-1] += len(dataset) - sum(lengths)

    return random_split(dataset, lengths, generator=torch.Generator().manual_seed(seed))


def partition_data_non_iid_label(dataset, num_clients: int, classes_per_client: int = 2,
                                 seed: int = 42):
    """
    Non-IID partition: each client has imbalanced class distribution.
    Each client has all classes, but dominated by a few (label skew).
    """
    import numpy as np
    from torch.utils.data import Subset

    # Get all labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        # Extract labels by iterating (slower but works for all datasets)
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(labels))
    np.random.seed(seed)
    
    # Group indices by class
    class_indices = {i: np.where(labels == i)[0] for i in range(num_classes)}
    
    # Shuffle each class's indices
    for indices in class_indices.values():
        np.random.shuffle(indices)
    
    client_datasets = []
    
    for client_id in range(num_clients):
        client_indices = []
        
        # Assign primary classes to this client (they get 80% of samples)
        primary_classes = np.random.choice(num_classes, classes_per_client, replace=False)
        
        # For primary classes: give this client a large portion
        for cls in primary_classes:
            cls_idx = class_indices[cls]
            # Take 80% / num_clients samples from this class
            samples_per_client = len(cls_idx) // num_clients
            main_portion = int(samples_per_client * 0.8)
            start_idx = (client_id * samples_per_client) % len(cls_idx)
            end_idx = min(start_idx + main_portion, len(cls_idx))
            client_indices.extend(cls_idx[start_idx:end_idx])
        
        # For other classes: give this client a small portion (for generalization)
        other_classes = [c for c in range(num_classes) if c not in primary_classes]
        for cls in other_classes:
            cls_idx = class_indices[cls]
            # Take only 20% / num_clients samples from other classes
            samples_per_client = len(cls_idx) // num_clients
            small_portion = max(int(samples_per_client * 0.2), 10)  # At least 10 samples
            start_idx = (client_id * samples_per_client) % len(cls_idx)
            end_idx = min(start_idx + small_portion, len(cls_idx))
            client_indices.extend(cls_idx[start_idx:end_idx])
        
        # Shuffle and create subset
        client_indices = np.array(client_indices)
        np.random.shuffle(client_indices)
        client_datasets.append(Subset(dataset, client_indices))
    
    return client_datasets


def partition_data_non_iid_dirichlet(dataset, num_clients: int, alpha: float = 0.5,
                                     seed: int = 42):
    """
    Non-IID partition using Dirichlet distribution.
    Lower alpha = more non-IID (realistic federated scenario).
    """
    import numpy as np
    from torch.utils.data import Subset

    # Get all labels
    if hasattr(dataset, 'targets'):
        labels = np.array(dataset.targets)
    elif hasattr(dataset, 'labels'):
        labels = np.array(dataset.labels)
    else:
        labels = np.array([dataset[i][1] for i in range(len(dataset))])

    num_classes = len(np.unique(labels))
    np.random.seed(seed)
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    client_indices = [[] for _ in range(num_clients)]
    
    # For each class, distribute samples to clients according to Dirichlet distribution
    for cls in range(num_classes):
        cls_indices = np.where(labels == cls)[0]
        np.random.shuffle(cls_indices)
        
        # Split according to Dirichlet proportions
        proportions = label_distribution[cls]
        proportions = proportions / proportions.sum()  # Normalize
        splits = (proportions * len(cls_indices)).astype(int)
        splits[-1] = len(cls_indices) - splits[:-1].sum()  # Handle rounding
        
        # Distribute to clients
        idx = 0
        for client_id, split in enumerate(splits):
            if split > 0:
                client_indices[client_id].extend(cls_indices[idx:idx + split])
                idx += split
    
    # Create subsets
    client_datasets = []
    for indices in client_indices:
        if len(indices) > 0:
            client_datasets.append(Subset(dataset, indices))
        else:
            # If a client has no data, give it a small random sample
            random_indices = np.random.choice(len(dataset), 10, replace=False)
            client_datasets.append(Subset(dataset, random_indices))
    
    return client_datasets


def _site_group_arrays(dataset):
    """Per-sample site and specimen group of a ListImageDataset or a Subset of one."""
    from torch.utils.data import Subset
    idx = None
    base = dataset
    while isinstance(base, Subset):
        idx = np.asarray(base.indices) if idx is None else np.asarray(base.indices)[idx]
        base = base.dataset
    if getattr(base, "sites", None) is None or getattr(base, "groups", None) is None:
        raise ValueError("natural partitioning needs a dataset with sites and groups "
                         "(KIDNEY_MULTISITE, data version v2)")
    if idx is None:
        idx = np.arange(len(base))
    return base.sites[idx], base.groups[idx]


def partition_data_natural(dataset, num_clients: int, seed: int = 42):
    """Natural multi-site partition (KIDNEY_MULTISITE).

    Clients never mix sites: each site gets a number of clients proportional to
    its training size (largest remainder, at least one), and its specimens are
    dealt whole, in random order, to the currently smallest client of that site.
    Label skew is therefore the one the sites really have, not a simulated one.
    """
    from torch.utils.data import Subset
    sites, groups = _site_group_arrays(dataset)
    site_names = sorted(set(sites.tolist()))
    sizes = np.array([np.sum(sites == s) for s in site_names], dtype=float)
    quota = sizes / sizes.sum() * num_clients
    n_per_site = np.maximum(np.floor(quota).astype(int), 1)
    while n_per_site.sum() < num_clients:
        n_per_site[np.argmax(quota - n_per_site)] += 1
    while n_per_site.sum() > num_clients:
        n_per_site[np.argmax(n_per_site)] -= 1
    rng = np.random.RandomState(seed)
    client_indices = []
    for s, n_s in zip(site_names, n_per_site):
        site_idx = np.where(sites == s)[0]
        by_group = {}
        for i in site_idx:
            by_group.setdefault(groups[i], []).append(i)
        keys = sorted(by_group)
        rng.shuffle(keys)
        bins = [[] for _ in range(n_s)]
        for g in keys:
            min(bins, key=len).extend(by_group[g])
        client_indices.extend(bins)
        print(f"  - Site {s}: {len(site_idx)} samples, {len(keys)} specimens -> "
              f"{n_s} clients of sizes {[len(b) for b in bins]}")
    return [Subset(dataset, np.array(sorted(b))) for b in client_indices]


def _extract_labels(dataset):
    """Return an np.array of integer labels for a dataset (fast paths first)."""
    if hasattr(dataset, 'targets'):
        return np.array(dataset.targets)
    if hasattr(dataset, 'labels'):
        return np.array(dataset.labels)
    return np.array([dataset[i][1] for i in range(len(dataset))])


def carve_server_root(train_dataset, per_class, num_classes, seed, site=""):
    """Carve a class-balanced clean server root set out of the training data.

    Used by reference-based robust baselines (RFLPA/FLTrust, BOBA) that assume a
    small clean labeled set on the server. The carved samples are REMOVED from the
    pool used to build client partitions, so there is no client/server leakage.

    Returns ``(remaining_subset, root_subset, root_labels)`` where
    ``remaining_subset`` is a ``Subset`` with a ``.targets`` attribute set (so the
    downstream partitioners keep their fast label path), ``root_subset`` is the
    balanced server set, and ``root_labels`` are its integer labels (for building
    per-class references, e.g. BOBA vertices).

    ``site`` (KIDNEY_MULTISITE only) restricts the root set to one acquisition
    site, i.e. a server that only holds data from one hospital; classes that site
    lacks are then simply absent from the root set.
    """
    from torch.utils.data import Subset
    labels = _extract_labels(train_dataset)
    rng = np.random.RandomState(seed)
    root_idx, keep_idx = [], []
    in_site = np.ones(len(labels), dtype=bool)
    if site:
        in_site = _site_group_arrays(train_dataset)[0] == site
    for c in range(num_classes):
        idx_c = np.where((labels == c) & in_site)[0]
        rng.shuffle(idx_c)
        take = min(per_class, len(idx_c))
        root_idx.extend(idx_c[:take].tolist())
        keep_idx.extend(idx_c[take:].tolist())
    if site:  # samples of the other sites all stay with the clients
        keep_idx.extend(np.where(~in_site)[0].tolist())
    rng.shuffle(root_idx)
    keep_idx = sorted(keep_idx)
    remaining = Subset(train_dataset, keep_idx)
    remaining.targets = [int(labels[i]) for i in keep_idx]  # keep partitioners fast
    root_subset = Subset(train_dataset, root_idx)
    root_labels = np.array([int(labels[i]) for i in root_idx])
    return remaining, root_subset, root_labels


def load_data(dataset_name: str = "EMNIST", num_clients: int = 10, batch_size: int = 32,
              distribution: str = "iid", classes_per_client: int = 2,
              dirichlet_alpha: float = 0.5,
              seed: int = 42,
              num_workers: int = 0,
              server_root_per_class: int = 0, server_root_mode: str = "clean",
              server_root_site: str = "", server_root_severity: int = 3) -> Tuple[List[DataLoader], List[DataLoader], DataLoader, Dict]:
    """
    Load and partition the specified dataset for federated learning.
    
    Args:
        dataset_name: Name of the dataset ("EMNIST", "MNIST", "CIFAR10")
        num_clients: Number of clients to partition the data into
        batch_size: Batch size for data loaders
        distribution: Data distribution type ("iid", "non-iid-label", "non-iid-dirichlet")
        classes_per_client: For non-iid-label, number of classes per client
        dirichlet_alpha: For non-iid-dirichlet, concentration parameter
    
    Returns:
        Tuple of (train_loaders, val_loaders, test_loader, dataset_info)
    """
    # Load dataset
    train_dataset, test_dataset, dataset_info = load_dataset(dataset_name)

    # Optionally carve a clean, class-balanced server root set (for reference-based
    # robust baselines such as RFLPA/FLTrust and BOBA). Removed from the client pool.
    server_root = None
    if server_root_per_class and server_root_per_class > 0:
        train_dataset, root_subset, root_labels = carve_server_root(
            train_dataset, server_root_per_class, dataset_info["num_classes"], seed,
            site=server_root_site)
        if server_root_mode == "corrupted":
            # The server's own reference data are degraded like a corrupted client's.
            root_subset = CorruptedDataset(root_subset, severity=server_root_severity,
                                           seed=seed + 10007)
        elif server_root_mode != "clean":
            raise ValueError(f"unknown SERVER_ROOT_MODE {server_root_mode!r}")
        server_root = {
            "loader": DataLoader(root_subset, batch_size=batch_size, shuffle=True,
                                 num_workers=num_workers, pin_memory=True),
            "dataset": root_subset,
            "labels": root_labels,
        }
        print(f"  - Server root set: {len(root_subset)} samples "
              f"({server_root_per_class}/class) carved out for reference-based defense"
              f" [mode={server_root_mode}, site={server_root_site or 'all'}]")

    # Partition training data based on distribution type
    if distribution == "iid":
        print(f"  - Distribution: IID (random split)")
        client_datasets = partition_data_iid(train_dataset, num_clients, seed=seed)
    elif distribution == "non-iid-label":
        print(f"  - Distribution: Non-IID (label skew, {classes_per_client} classes per client)")
        client_datasets = partition_data_non_iid_label(train_dataset, num_clients, classes_per_client, seed=seed)
    elif distribution == "non-iid-dirichlet":
        print(f"  - Distribution: Non-IID (Dirichlet alpha={dirichlet_alpha})")
        client_datasets = partition_data_non_iid_dirichlet(train_dataset, num_clients, dirichlet_alpha, seed=seed)
    elif distribution == "natural":
        print(f"  - Distribution: natural (clients = specimen-disjoint shares of real sites)")
        client_datasets = partition_data_natural(train_dataset, num_clients, seed=seed)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
    # Apply corruptions to selected clients (if enabled)
    import config as cfg
    corrupted_client_ids = []
    if getattr(cfg, 'CORRUPT_CLIENTS', False):
        corrupt_ratio = getattr(cfg, 'CORRUPT_CLIENT_RATIO', 0.3)
        corruption_severity = getattr(cfg, 'CORRUPTION_SEVERITY', 3)
        poison_mode = getattr(cfg, 'POISON_MODE', 'image')
        flip_ratio = getattr(cfg, 'LABEL_FLIP_RATIO', 1.0)
        client_datasets, corrupted_client_ids = apply_corruptions_to_clients(
            client_datasets,
            corrupt_ratio=corrupt_ratio,
            severity=corruption_severity,
            seed=seed,
            poison_mode=poison_mode,
            num_classes=dataset_info["num_classes"],
            flip_ratio=flip_ratio,
        )
    
    # Create train and validation loaders for each client
    train_loaders = []
    val_loaders = []
    
    for client_dataset in client_datasets:
        # Split each client's data into train/val (90/10)
        len_val = len(client_dataset) // 10
        len_train = len(client_dataset) - len_val
        
        ds_train, ds_val = random_split(
            client_dataset,
            [len_train, len_val],
            generator=torch.Generator().manual_seed(seed)
        )
        
        train_loaders.append(DataLoader(
            ds_train, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True))
        val_loaders.append(DataLoader(
            ds_val, batch_size=batch_size,
            num_workers=num_workers, pin_memory=True))

    # Create centralized test loader
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True)

    dataset_info["server_root"] = server_root  # None unless server_root_per_class>0
    # Class names and (multi-site only) the site of every test sample, in test
    # loader order, for per-class and per-site clinical metrics.
    dataset_info["class_names"] = list(getattr(test_dataset, "classes", []))
    test_sites = getattr(test_dataset, "sites", None)
    dataset_info["test_sites"] = test_sites.tolist() if test_sites is not None else None
    dataset_info["data_version"] = data_version()

    return train_loaders, val_loaders, test_loader, dataset_info

