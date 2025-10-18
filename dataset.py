import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, List, Dict


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
            "input_size": 224,
            "mean": [0.5],
            "std": [0.5]
        },
        "MNIST": {
            "num_classes": 10,
            "num_channels": 1,
            "input_size": 224,
            "mean": [0.5],
            "std": [0.5]
        },
        "CIFAR10": {
            "num_classes": 10,
            "num_channels": 3,
            "input_size": 224,
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
    
    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")
    
    return train_dataset, test_dataset, dataset_info


def partition_data_iid(dataset, num_clients: int):
    """IID partition: randomly split data equally among clients."""
    partition_size = len(dataset) // num_clients
    lengths = [partition_size] * num_clients
    lengths[-1] += len(dataset) - sum(lengths)
    
    return random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))


def partition_data_non_iid_label(dataset, num_clients: int, classes_per_client: int = 2):
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
    np.random.seed(42)
    
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


def partition_data_non_iid_dirichlet(dataset, num_clients: int, alpha: float = 0.5):
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


def load_data(dataset_name: str = "EMNIST", num_clients: int = 10, batch_size: int = 32,
              distribution: str = "iid", classes_per_client: int = 2, 
              dirichlet_alpha: float = 0.5) -> Tuple[List[DataLoader], List[DataLoader], DataLoader, Dict]:
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
    
    # Partition training data based on distribution type
    if distribution == "iid":
        print(f"  - Distribution: IID (random split)")
        client_datasets = partition_data_iid(train_dataset, num_clients)
    elif distribution == "non-iid-label":
        print(f"  - Distribution: Non-IID (label skew, {classes_per_client} classes per client)")
        client_datasets = partition_data_non_iid_label(train_dataset, num_clients, classes_per_client)
    elif distribution == "non-iid-dirichlet":
        print(f"  - Distribution: Non-IID (Dirichlet α={dirichlet_alpha})")
        client_datasets = partition_data_non_iid_dirichlet(train_dataset, num_clients, dirichlet_alpha)
    else:
        raise ValueError(f"Unknown distribution: {distribution}")
    
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
            generator=torch.Generator().manual_seed(42)
        )
        
        train_loaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        val_loaders.append(DataLoader(ds_val, batch_size=batch_size))
    
    # Create centralized test loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loaders, val_loaders, test_loader, dataset_info

