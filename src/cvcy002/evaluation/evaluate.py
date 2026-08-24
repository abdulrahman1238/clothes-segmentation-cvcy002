# src/cvcy002/evaluation/evaluate.py

import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from torch.utils.data import DataLoader

from .metrics import SegmentationMetrics
from ..data.LIPpreprocessing import COLOR_MAP

def save_visualizations(
    images: torch.Tensor, 
    gt_masks: torch.Tensor, 
    pred_masks: torch.Tensor, 
    image_names: list, 
    save_dir: str, 
    num_samples: int = 5
) -> None:
    """
    Unnormalizes images, colorizes masks, and saves a side-by-side grid.
    """
    # 1. Unnormalize images (ImageNet stats)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1).permute(0, 2, 3, 1).numpy() # (B, H, W, 3)
    
    gt_masks = gt_masks.numpy()
    pred_masks = pred_masks.numpy()
    
    # 2. Helper to colorize masks
    def colorize(mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color in COLOR_MAP.items():
            color_mask[mask == cls] = color
        return color_mask

    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
        
    for i in range(num_samples):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f"Input: {image_names[i]}")
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(colorize(gt_masks[i]))
        axes[i, 1].set_title("Ground Truth")
        axes[i, 1].axis("off")
        
        axes[i, 2].imshow(colorize(pred_masks[i]))
        axes[i, 2].set_title("Prediction")
        axes[i, 2].axis("off")
        
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "eval_visualizations.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Visualizations saved to: {save_path}")


def run_evaluation(
    model: torch.nn.Module,
    dataloader: DataLoader,
    config: Dict,
    device: torch.device,
    save_vis: bool = True
) -> Dict:
    """
    Main evaluation loop. Runs inference, calculates metrics, and saves artifacts.
    """
    print("Starting Evaluation...")
    model.eval()
    
    num_classes = config["model"]["num_classes"]
    class_names = config["model"]["class_names"]
    
    # Initialize Metrics Calculator
    metrics_calc = SegmentationMetrics(num_classes, class_names, device)
    use_amp = (device.type == 'cuda')
    
    # Directories for artifacts
    vis_dir = config["paths"]["visualization_dir"]
    metrics_dir = config["paths"]["metrics_dir"]
    os.makedirs(vis_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    
    # Storage for visualizations (we only need the first batch)
    vis_images, vis_gt, vis_pred, vis_names = [], [], [], []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            names = batch["image_name"]
            
            # Forward pass with AMP
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(images)
                
            preds = torch.argmax(outputs, dim=1)
            
            # Update metrics
            metrics_calc.update(preds, masks)
            
            # Store first batch for visualization
            if save_vis and i == 0:
                vis_images.append(images.cpu())
                vis_gt.append(masks.cpu())
                vis_pred.append(preds.cpu())
                vis_names.extend(names)
                
    # 1. Compute Final Metrics
    final_metrics = metrics_calc.compute()
    
    # 2. Save Visualizations
    if save_vis and vis_images:
        imgs = torch.cat(vis_images, dim=0)
        gt_masks = torch.cat(vis_gt, dim=0)
        pred_masks = torch.cat(vis_pred, dim=0)
        save_visualizations(imgs, gt_masks, pred_masks, vis_names, vis_dir, num_samples=5)
        
    # 3. Save Metrics to JSON
    metrics_path = os.path.join(metrics_dir, "evaluation_results.json")
    with open(metrics_path, "w") as f:
        json.dump(final_metrics, f, indent=4)
    print(f"Metrics saved to: {metrics_path}")
    
    # 4. Print Summary
    print("-" * 40)
    print(f"Final mIoU:        {final_metrics['miou']:.4f}")
    print(f"Pixel Accuracy:    {final_metrics['pixel_acc']:.4f}")
    print(f"Macro F1-Score:    {final_metrics['macro_f1']:.4f}")
    print(f"Per Class IOU:    {final_metrics['per_class_iou']:.4f}")
    print(f"Per Class F1-Score:    {final_metrics['per_class_f1']:.4f}")
    print("-" * 40)
    
    return final_metrics