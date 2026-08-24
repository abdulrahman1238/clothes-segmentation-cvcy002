# src/cvcy002/models/deeplabv3plus.py

import os
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from typing import Union, Dict

class DeepLabV3PlusModel(nn.Module):
    """
    SMP DeepLabV3+ model.
    model creation, checkpoint loading, and parameter inspection.
    """
    def __init__(self, config: Dict):
        super().__init__()
        model_cfg = config["model"]
        
        backbone = model_cfg.get("backbone", "resnet50")
        num_classes = model_cfg.get("num_classes", 2)
        in_channels = model_cfg.get("in_channels", 3)
        pretrained = model_cfg.get("pretrained", True)
        
        # SMP uses "imagenet" for pretrained weights, or None for random initialization
        encoder_weights = "imagenet" if pretrained else None
        
        print(f"Building DeepLabV3+ | Backbone: {backbone} | Classes: {num_classes} | Pretrained: {pretrained}")
        
        # Initialize the core SMP model
        self.model = smp.DeepLabV3Plus(
            encoder_name=backbone,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=num_classes,
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Standard PyTorch forward pass."""
        return self.model(x)
        
    def load_checkpoint(self, checkpoint_path: str, device: Union[str, torch.device] = "cpu") -> None:
        """
        Safely loads model weights from a checkpoint file.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
        print(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Handle both raw state_dict and dicts containing 'model_state_dict' 
        # (best practice when saving optimizer state alongside the model)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        print("Checkpoint loaded successfully.")
        
    def get_info(self) -> None:
        """Calculates and prints the number of parameters in the model."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("-" * 45)
        print(f"Model Architecture: {self.__class__.__name__}")
        print(f"Total Parameters:     {total_params:>12,}")
        print(f"Trainable Parameters: {trainable_params:>12,}")
        print("-" * 45)