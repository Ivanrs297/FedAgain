"""
Simple CSV logger for federated learning experiments.
Logs experiment configuration and results (accuracy, precision, recall, F1).
"""

import csv
import os
from datetime import datetime
from pathlib import Path
import numpy as np


class ExperimentLogger:
    """Logger for federated learning experiments."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize experiment logger.
        
        Args:
            results_dir: Directory to save results CSV
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.csv_path = self.results_dir / "experiments.csv"
        self._init_csv()
    
    def _init_csv(self):
        """Initialize CSV file with headers if it doesn't exist."""
        if not self.csv_path.exists():
            headers = [
                "timestamp",
                "date",
                "model",
                "strategy",
                "dataset",
                "data_distribution",
                "num_clients",
                "num_rounds",
                "local_epochs",
                "batch_size",
                "learning_rate",
                "final_accuracy",
                "final_precision",
                "final_recall",
                "final_f1",
                "use_pretrained",
                "corrupt_clients",
                "corrupt_ratio",
                "corruption_severity"
            ]
            
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
    
    def log_experiment(self, config_dict: dict, metrics: dict):
        """
        Log experiment configuration and results to CSV.
        
        Args:
            config_dict: Dictionary with experiment configuration
            metrics: Dictionary with final metrics (accuracy, precision, recall, f1)
        """
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
        date = now.strftime("%Y-%m-%d")
        
        row = [
            timestamp,
            date,
            config_dict.get("model", "ResNet18"),
            config_dict.get("strategy", "Unknown"),
            config_dict.get("dataset", "Unknown"),
            config_dict.get("data_distribution", "Unknown"),
            config_dict.get("num_clients", 0),
            config_dict.get("num_rounds", 0),
            config_dict.get("local_epochs", 0),
            config_dict.get("batch_size", 0),
            config_dict.get("learning_rate", 0.0),
            metrics.get("accuracy", 0.0),
            metrics.get("precision", 0.0),
            metrics.get("recall", 0.0),
            metrics.get("f1", 0.0),
            config_dict.get("use_pretrained", True),
            config_dict.get("corrupt_clients", False),
            config_dict.get("corrupt_ratio", 0.0),
            config_dict.get("corruption_severity", 0)
        ]
        
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)
        
        print(f"\n[OK] Results saved to: {self.csv_path}")


def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                   num_classes: int) -> dict:
    """
    Compute precision, recall, and F1 score (macro-averaged).
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes
    
    Returns:
        Dictionary with precision, recall, f1, and accuracy
    """
    # Compute per-class metrics
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    
    for cls in range(num_classes):
        # True positives, false positives, false negatives
        tp = np.sum((y_true == cls) & (y_pred == cls))
        fp = np.sum((y_true != cls) & (y_pred == cls))
        fn = np.sum((y_true == cls) & (y_pred != cls))
        
        # Precision
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        precision_per_class.append(precision)
        
        # Recall
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        recall_per_class.append(recall)
        
        # F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_per_class.append(f1)
    
    # Macro average
    macro_precision = np.mean(precision_per_class)
    macro_recall = np.mean(recall_per_class)
    macro_f1 = np.mean(f1_per_class)
    
    # Accuracy
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    return {
        "accuracy": accuracy,
        "precision": macro_precision,
        "recall": macro_recall,
        "f1": macro_f1
    }

