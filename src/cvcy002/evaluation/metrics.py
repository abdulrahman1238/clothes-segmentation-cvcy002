# src/cvcy002/evaluation/metrics.py

import torch
from typing import Dict, List, Optional

class SegmentationMetrics:
    """
    Standalone utility to calculate segmentation metrics using a Global Confusion Matrix.
    Can be used by both the Trainer and the Evaluation script.
    """
    def __init__(
        self, 
        num_classes: int, 
        class_names: Optional[List[str]] = None, 
        device: str = "cpu"
    ):
        self.num_classes = num_classes
        self.class_names = class_names 
        self.device = device
        
        # Initialize Global Confusion Matrix
        self.confusion_matrix = torch.zeros(
            (num_classes, num_classes), 
            dtype=torch.int64, 
            device=device
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Updates the confusion matrix with a new batch of predictions and targets.
        """
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # Filter out any invalid labels (e.g., ignore_index)
        mask = (targets >= 0) & (targets < self.num_classes)
        preds = preds[mask]
        targets = targets[mask]
        
        # Map (target, pred) pairs to 1D indices and count them
        indices = targets * self.num_classes + preds
        counts = torch.bincount(indices, minlength=self.num_classes**2)
        
        self.confusion_matrix += counts.view(self.num_classes, self.num_classes)

    def compute(self) -> Dict:
        """
        Calculates all metrics from the accumulated confusion matrix.
        Returns a dictionary formatted for easy JSON serialization.
        """
        cm = self.confusion_matrix.float()
        
        # True Positives, False Positives, False Negatives
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        
        # 1. IoU and mIoU
        iou = tp / (tp + fp + fn + 1e-7)
        miou = iou.mean().item()
        
        # 2. Pixel Accuracy
        pixel_acc = tp.sum() / (cm.sum() + 1e-7)
        
        # 3. Precision, Recall, F1
        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        
        # Format per-class metrics into dictionaries for clean JSON output
        per_class_iou = {name: iou[i].item() for i, name in enumerate(self.class_names)}
        per_class_f1 = {name: f1[i].item() for i, name in enumerate(self.class_names)}
        
        return {
            "miou": miou,
            "pixel_acc": pixel_acc.item(),
            "macro_precision": precision.mean().item(),
            "macro_recall": recall.mean().item(),
            "macro_f1": f1.mean().item(),
            "per_class_iou": per_class_iou,
            "per_class_f1": per_class_f1,
        }

    def reset(self) -> None:
        """Zeros the confusion matrix for a fresh evaluation run."""
        self.confusion_matrix.zero_()