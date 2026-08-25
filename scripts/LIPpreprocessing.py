# scripts/LIPpreprocessing.py

import yaml
import os
import numpy as np
import cv2
from collections import Counter

from cvcy002.data import LIPDataset, visualize_sample, LIP_TO_2_CLASS

def main():
    with open("configs/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    data_root = config["paths"]["data_dir"]
    print(f"Checking dataset at: {data_root}")

    # 1. Verify directory structure
    required_dirs = [
        config["paths"]["train_image_dir"],
        config["paths"]["train_mask_dir"],
        config["paths"]["val_image_dir"],
        config["paths"]["val_mask_dir"]
    ]
    for d in required_dirs:
        full_path = os.path.join(data_root, d)
        if not os.path.exists(full_path):
            print(f"❌ MISSING DIRECTORY: {full_path}")
            return
    print("✅ All required directories found.")

    # 2. Verify split files
    for split in [config["paths"]["train_split_file"], config["paths"]["val_split_file"]]:
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path):
            print(f"❌ MISSING SPLIT FILE: {split_path}")
            return
        with open(split_path, 'r') as f:
            count = len([line for line in f.readlines() if line.strip()])
        print(f"✅ Found {split} with {count} image IDs.")

    # 3. Analyze class distribution in a small sample
    print("\nAnalyzing class distribution in first 10 training masks...")
    train_mask_dir = os.path.join(data_root, config["paths"]["train_mask_dir"])
    split_file = os.path.join(data_root, config["paths"]["train_split_file"])
    
    with open(split_file, 'r') as f:
        ids = [line.strip() for line in f.readlines() if line.strip()][:10]
        
    class_counts = Counter()
    for img_id in ids:
        mask_path = os.path.join(train_mask_dir, f"{img_id}.png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mapped_mask = LIP_TO_2_CLASS[mask]
            unique, counts = np.unique(mapped_mask, return_counts=True)
            for cls, count in zip(unique, counts):
                class_counts[cls] += count
                
    total_pixels = sum(class_counts.values())
    print("Class Distribution (Mapped to 2 Classes):")
    names = {0: "Background", 1: "Clothes"}
    for cls in sorted(class_counts.keys()):
        pct = (class_counts[cls] / total_pixels) * 100
        print(f"  Class {cls} ({names[cls]}): {pct:.2f}%")

    # 4. Visualize a sample to ensure mapping is correct
    print("\nGenerating visualization of sample 0...")
    dataset = LIPDataset(
        image_dir=os.path.join(data_root, config["paths"]["train_image_dir"]),
        mask_dir=train_mask_dir,
        split_file=split_file
    )
    visualize_sample(dataset, config, idx=0, save_path="outputs/visualizations/data_check_sample.png")
    print("✅ Data verification complete!")

if __name__ == "__main__":
    main()