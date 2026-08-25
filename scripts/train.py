# scripts/train.py

import argparse
import yaml
import torch
import os
import random
import numpy as np

from cvcy002.data import LIPDataset, get_transforms
from cvcy002.models import DeepLabV3PlusModel
from cvcy002.training import Trainer, CombinedLoss
from torch.utils.data import DataLoader

def set_seed(seed: int):
    """Sets seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="Train DeepLabV3+ for Clothes Segmentation")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config file")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # 2. Load Config & Set Seed
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    set_seed(config["system"]["seed"])

    # 3. Setup Device
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 4. Build Datasets & DataLoaders
    data_root = config["paths"]["data_dir"]
    
    train_dataset = LIPDataset(
        image_dir=os.path.join(data_root, config["paths"]["train_image_dir"]),
        mask_dir=os.path.join(data_root, config["paths"]["train_mask_dir"]),
        split_file=os.path.join(data_root, config["paths"]["train_split_file"]),
        transform=get_transforms("train", config)
    )
    val_dataset = LIPDataset(
        image_dir=os.path.join(data_root, config["paths"]["val_image_dir"]),
        mask_dir=os.path.join(data_root, config["paths"]["val_mask_dir"]),
        split_file=os.path.join(data_root, config["paths"]["val_split_file"]),
        transform=get_transforms("val", config)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True, 
        num_workers=config["training"]["num_workers"], pin_memory=config["training"]["pin_memory"]
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["training"]["batch_size"], shuffle=False, 
        num_workers=config["training"]["num_workers"], pin_memory=config["training"]["pin_memory"]
    )

    # 5. Build Model, Optimizer, Scheduler, Criterion
    model = DeepLabV3PlusModel(config).to(device)
    model.get_info()

    opt_cfg = config["training"]["optimizer"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])

    sch_cfg = config["training"]["scheduler"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=sch_cfg["T_max"], eta_min=sch_cfg["eta_min"])

    criterion = CombinedLoss(config).to(device)

    # 6. Initialize Trainer & Handle Resume
    trainer = Trainer(
        model=model, train_loader=train_loader, val_loader=val_loader,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        device=device, config=config
    )

    if args.resume:
        trainer.resume_from_checkpoint(args.resume)

    # 7. Start Training
    trainer.fit()

if __name__ == "__main__":
    main()