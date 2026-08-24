# src/cvcy002/training/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class DiceLoss(nn.Module):
    """
    Computes the Dice loss.
    Optimized for segmentation by ignoring the background class to focus.
    """
    def __init__(self, smooth: float = 1e-7, ignore_background: bool = True):
        super().__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model predictions of shape (B, C, H, W)
            targets: Ground truth masks of shape (B, H, W) containing class indices
        Returns:
            Scalar Dice loss tensor
        """
        # 1. Apply softmax to get probabilities (B, C, H, W)
        probs = F.softmax(logits, dim=1)
        
        # 2. One-hot encode targets to match logits shape: (B, H, W) -> (B, C, H, W)
        num_classes = logits.shape[1]
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).permute(0, 3, 1, 2).float()
        
        # 3. Optionally ignore background (class 0) to focus on foreground
        if self.ignore_background:
            probs = probs[:, 1:, :, :]          # Shape: (B, C-1, H, W)
            targets_one_hot = targets_one_hot[:, 1:, :, :] # Shape: (B, C-1, H, W)
        
        # 4. Calculate intersection and union across spatial dimensions (H, W)
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        
        # 5. Calculate Dice coefficient with smoothing for numerical stability
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # 6. Return mean Dice loss across batch and remaining classes
        # Loss is 1 - Dice, so minimizing loss maximizes Dice (overlap)
        return 1.0 - dice.mean()


class CombinedLoss(nn.Module):
    """
    Combines Weighted Cross-Entropy Loss and Foreground Dice Loss.
    CE handles pixel-wise classification confidence.
    Dice handles region-level overlap and class imbalance.
    """
    def __init__(self, config: Dict):
        super().__init__()
        
        # Extract loss configuration, with sensible defaults for LIP 3-class mapping
        loss_cfg = config.get("training", {}).get("loss", {})
        
        # Weights for [Background, Person, Clothes]
        # Background gets lower weight (0.2) because it's dominant and easy to learn.
        # Person and Clothes get higher weights (0.4 each) to force the model to care about them.
        ce_weights_list = loss_cfg.get("ce_weights", [0.3, 0.7])
        
        # Relative weighting between the two loss components
        self.ce_weight = loss_cfg.get("ce_weight", 0.5)
        self.dice_weight = loss_cfg.get("dice_weight", 0.5)
        
        # Register weights as a buffer so they automatically move to the correct device (CPU/GPU)
        self.register_buffer('ce_weights', torch.tensor(ce_weights_list, dtype=torch.float32))
        
        # Cross-Entropy expects raw logits and class indices. It applies log-softmax internally.
        self.ce_loss = nn.CrossEntropyLoss(weight=self.ce_weights)
        
        # Our custom Dice loss (ignores background by default)
        self.dice_loss = DiceLoss(ignore_background=True)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Raw model predictions of shape (B, C, H, W)
            targets: Ground truth masks of shape (B, H, W)
        Returns:
            Combined scalar loss tensor
        """
        loss_ce = self.ce_loss(logits, targets)
        loss_dice = self.dice_loss(logits, targets)
        
        return self.ce_weight * loss_ce + self.dice_weight * loss_dice