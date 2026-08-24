# scripts/predict.py

import argparse
import yaml
import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from cvcy002.models import DeepLabV3PlusModel
from cvcy002.data.LIPpreprocessing import COLOR_MAP
import albumentations as A
from albumentations.pytorch import ToTensorV2

def get_inference_transform(config):
    """Transform for a single raw image (no ground truth mask)."""
    h, w = config["training"]["image_size"]
    return A.Compose([
        A.Resize(height=h, width=w),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def main():
    parser = argparse.ArgumentParser(description="Predict segmentation on a single personal image")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True, help="Path to your personal image")
    parser.add_argument("--output_path", type=str, default="outputs/predictions/result.png")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    model = DeepLabV3PlusModel(config).to(device)
    model.load_checkpoint(args.checkpoint, device)
    
    # 2. Load & Preprocess Image
    image_bgr = cv2.imread(args.image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not load image at {args.image_path}")
    
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    original_h, original_w = image_rgb.shape[:2]
    
    transform = get_inference_transform(config)
    augmented = transform(image=image_rgb)
    input_tensor = augmented["image"].unsqueeze(0).to(device)
    
    # 3. Inference
    model.eval()
    use_amp = (device.type == 'cuda')
    with torch.no_grad():
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(input_tensor)
            
    # Get class predictions and move to CPU
    preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
    
    # 4. Resize prediction back to original image resolution
    pred_resized = cv2.resize(preds, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    
    # 5. Colorize the mask
    color_mask = np.zeros((original_h, original_w, 3), dtype=np.uint8)
    for cls, color in COLOR_MAP.items():
        color_mask[pred_resized == cls] = color
        
    # 6. Blend mask with original image
    alpha = 0.6
    blended = cv2.addWeighted(image_rgb, alpha, color_mask, 1 - alpha, 0)
    
    # 7. Save Results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save the blended image (requires BGR for cv2)
    blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
    cv2.imwrite(args.output_path, blended_bgr)
    print(f"Blended prediction saved to: {args.output_path}")
    
    # Also save a side-by-side comparison for the report
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image_rgb); axes[0].set_title("Original Image"); axes[0].axis("off")
    axes[1].imshow(color_mask); axes[1].set_title("Predicted Mask (BG, Person, Clothes)"); axes[1].axis("off")
    axes[2].imshow(blended); axes[2].set_title("Blended Overlay"); axes[2].axis("off")
    plt.tight_layout()
    
    grid_path = args.output_path.replace(".png", "_grid.png")
    plt.savefig(grid_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Comparison grid saved to: {grid_path}")

if __name__ == "__main__":
    main()