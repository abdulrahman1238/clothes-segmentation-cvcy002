import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List

# -----------------------------------------------------------------------------
# Label Mapping Configuration
# -----------------------------------------------------------------------------
# LIP Original Labels (0-19):
# 0: Background, 1: Hat, 2: Hair, 3: Glove, 4: Sunglasses, 5: Upper-clothes, 
# 6: Dress, 7: Coat, 8: Socks, 9: Pants, 10: Jumpsuits, 11: Shoes, 12: Bag, 
# 13: Scarf, 14: Skirt, 15: Face, 16: Left-arm, 17: Right-arm, 18: Left-leg, 19: Right-leg

# Mapping to 3 Classes: 0=Background, 1=Person, 2=Clothes
#LIP_TO_3_CLASS = np.zeros(256, dtype=np.uint8)
#LIP_TO_3_CLASS[0] = 0  # Background
#LIP_TO_3_CLASS[[1, 2, 3, 4, 13, 14, 15, 16, 17, 18, 19]] = 1  # Person parts
#LIP_TO_3_CLASS[[5, 6, 7, 8, 9, 10, 11, 12]] = 2  # Clothes parts

LIP_TO_2_CLASS = np.zeros(256, dtype=np.uint8)
LIP_TO_2_CLASS[0] = 0  # Background
LIP_TO_2_CLASS[[2, 4, 13, 14, 15, 16, 17]] = 0  # Person -> Background
LIP_TO_2_CLASS[[1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 18, 19]] = 1 

# Color map for visualization (RGB)
COLOR_MAP = {
    0: (0, 0, 0),       # Background: Black
   # 1: (0, 128, 255),   # Person: Light Blue
    1: (0, 255, 0)      # Clothes: Green
}

# -----------------------------------------------------------------------------
# Dataset Class
# -----------------------------------------------------------------------------
class LIPDataset(Dataset):
    def __init__(
        self, 
        image_dir: str, 
        mask_dir: str, 
        transform: Optional[A.Compose] = None,
        split_file: Optional[str] = None
    ):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        # 1. If a split .txt file is provided, use it (Recommended)
        if split_file and os.path.exists(split_file):
            with open(split_file, 'r') as f:
                # Read lines, strip whitespace/newlines, and add .jpg/.png extensions
                ids = [line.strip() for line in f.readlines() if line.strip()]
            
            # Assume images are .jpg and masks are .png (standard for LIP)
            self.image_files = [f"{img_id}.jpg" for img_id in ids]
            self.mask_files = [f"{img_id}.png" for img_id in ids]
            
        # 2. Fallback: If no .txt file, scan the directory
        else:
            self.image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
            self.mask_files = [f.rsplit('.', 1)[0] + '.png' for f in self.image_files]
            
        # Sanity check: ensure image and mask files match in length
        assert len(self.image_files) == len(self.mask_files), "Mismatch between images and masks!"
        
    def __len__(self) -> int:
        return len(self.image_files)
    
        
    def __getitem__(self, idx: int) -> dict:
        img_name = self.image_files[idx]
        mask_name = self.mask_files[idx]
        
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # 1. Load raw data
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if image is None or mask is None:
            raise FileNotFoundError(f"Could not load image or mask for {img_name}")
        
        # 2. Apply 20-to-3 class mapping ON THE RAW MASK
        #mask_mapped = LIP_TO_3_CLASS[mask]
        mask_mapped = LIP_TO_2_CLASS[mask]
        
        # 3. Apply Albumentations (Handles ALL resizing, cropping, etc.)
        if self.transform:
            augmented = self.transform(image=image, mask=mask_mapped)
            image = augmented["image"]
            mask_mapped = augmented["mask"]
        else:
            # Fallback if no transform is provided
            image = torch.tensor(image.transpose(2, 0, 1), dtype=torch.float32) / 255.0
            mask_mapped = torch.tensor(mask_mapped, dtype=torch.long)
            
        return {
            "image": image,
            "mask": mask_mapped,
            "image_name": img_name
        }

# -----------------------------------------------------------------------------
# Transform Factory
# -----------------------------------------------------------------------------
def get_transforms(split: str, config: dict) -> A.Compose:
    """Builds albumentations pipeline based on split and config."""
    h, w = config["training"]["image_size"]
    
    if split == "train":
        aug_config = config["augmentation"]["train"]
        return A.Compose([
            A.HorizontalFlip(p=aug_config["horizontal_flip"]),
            
            A.Affine(
                    scale=tuple(aug_config["affine"]["scale"]),
                    translate_percent=tuple(aug_config["affine"]["translate_percent"]),
                    rotate=tuple(aug_config["affine"]["rotate"]),
                    p=aug_config["affine"]["p"]),


            A.Resize(height=h, width=w),

            A.ColorJitter(
                brightness=aug_config["color_jitter"]["brightness"],
                contrast=aug_config["color_jitter"]["contrast"],
                saturation=aug_config["color_jitter"]["saturation"],
                hue=aug_config["color_jitter"]["hue"],
                p=aug_config["color_jitter"]["p"]),

            A.GaussNoise(
                var_limit=tuple(
                    aug_config["gaussian_noise"]["var_limit"]
                ),
                p=aug_config["gaussian_noise"]["p"],
            ),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])
    else: # val or test
        aug_config = config["augmentation"]["val"]
        return A.Compose([
            A.Resize(height=h, width=w),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])


# -----------------------------------------------------------------------------
# Visualization Helper
# -----------------------------------------------------------------------------
def visualize_sample(
    dataset: LIPDataset,
    config: dict, 
    idx: int, 
    save_path: Optional[str] = None
) -> None:
    """
    Visualizes a single sample BEFORE and AFTER preprocessing/augmentation.
    """
    img_name = dataset.image_files[idx]
    img_path = os.path.join(dataset.image_dir, img_name)
    mask_name = img_name.rsplit('.', 1)[0] + '.png'
    mask_path = os.path.join(dataset.mask_dir, mask_name)
    h, w = config["training"]["image_size"]
    
    # 1. Load RAW data
    raw_image = cv2.imread(img_path)
    raw_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
    raw_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Resize raw to target size for fair comparison
    raw_image = cv2.resize(raw_image, (w, h), interpolation=cv2.INTER_LINEAR)
    raw_mask = cv2.resize(raw_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    #raw_mask_mapped = LIP_TO_3_CLASS[raw_mask]
    raw_mask_mapped = LIP_TO_2_CLASS[raw_mask]
    
    # 2. Get AUGMENTED data (via __getitem__)
    sample = dataset[idx]
    aug_image = sample["image"].numpy().transpose(1, 2, 0)
    # Unnormalize for visualization
    aug_image = (aug_image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])) * 255
    aug_image = np.clip(aug_image, 0, 255).astype(np.uint8)
    aug_mask_mapped = sample["mask"].numpy()
    
    # 3. Create colorized masks
    def colorize_mask(mask: np.ndarray) -> np.ndarray:
        h, w = mask.shape
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        for cls, color in COLOR_MAP.items():
            color_mask[mask == cls] = color
        return color_mask

    raw_color_mask = colorize_mask(raw_mask_mapped)
    aug_color_mask = colorize_mask(aug_mask_mapped)
    
    # 4. Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0, 0].imshow(raw_image)
    axes[0, 0].set_title("Raw Image")
    axes[0, 0].axis("off")
    
    axes[0, 1].imshow(raw_color_mask)
    axes[0, 1].set_title("Raw Mask (Mapped to 2 Classes)\nBlack=BG, Green=Clothes")
    axes[0, 1].axis("off")
    
    axes[1, 0].imshow(aug_image)
    axes[1, 0].set_title("Augmented Image")
    axes[1, 0].axis("off")
    
    axes[1, 1].imshow(aug_color_mask)
    axes[1, 1].set_title("Augmented Mask")
    axes[1, 1].axis("off")
    
    plt.suptitle(f"Sample Visualization: {img_name}", fontsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150)
        print(f"Visualization saved to: {save_path}")
    else:
        plt.show()
    
    plt.close()