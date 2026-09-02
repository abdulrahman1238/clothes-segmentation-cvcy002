# scripts/evaluate.py

import argparse
import yaml
import torch
import os

from cvcy002.data import LIPDataset, get_transforms
from cvcy002.models import DeepLabV3PlusModel
from cvcy002.evaluation import run_evaluation
from torch.utils.data import DataLoader

def main():
    parser = argparse.ArgumentParser(description="Evaluate DeepLabV3+ Model")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load Validation/Test Dataset
    data_root = config["paths"]["data_dir"]
    val_dataset = LIPDataset(
        image_dir=os.path.join(data_root, config["paths"]["val_image_dir"]),
        mask_dir=os.path.join(data_root, config["paths"]["val_mask_dir"]),
        split_file=os.path.join(data_root, config["paths"]["val_split_file"]),
        transform=get_transforms("val", config)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, 
        num_workers=config["training"]["num_workers"], pin_memory=config["training"]["pin_memory"]
    )

    # Load Model & Checkpoint
    model = DeepLabV3PlusModel(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Extract the actual model weights from the checkpoint dictionary
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f" Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    else:
        state_dict = checkpoint


    model.load_state_dict(state_dict)
    model.to(device)
    model.get_info()

    # Run Evaluation
    run_evaluation(
        model=model,
        dataloader=val_loader,
        config=config,
        device=device,
        save_vis=True
    )

if __name__ == "__main__":
    main()
