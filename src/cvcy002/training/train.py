# src/cvcy002/training/train.py

import os
import time
import torch
import torch.nn as nn
from typing import Dict
from tqdm import tqdm
from ..evaluation.metrics import SegmentationMetrics

class Trainer:
    """
    Stateful Trainer class handling the training loop, validation, 
    metric tracking (Global Confusion Matrix), and checkpointing.
    Optimized for Kaggle with Mixed Precision (AMP).
    """
    def __init__(
        self, 
        model: nn.Module, 
        train_loader: torch.utils.data.DataLoader, 
        val_loader: torch.utils.data.DataLoader, 
        criterion: nn.Module, 
        optimizer: torch.optim.Optimizer, 
        scheduler: torch.optim.lr_scheduler._LRScheduler, 
        device: torch.device, 
        config: Dict
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        # Extract training parameters from config
        self.num_classes = config["model"]["num_classes"]
        self.class_names = config["model"]["class_names"]
        self.epochs = config["training"]["epochs"]
        self.patience = config["training"].get("early_stopping_patience", 5)
        self.checkpoint_dir = config["paths"]["checkpoint_dir"]
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        # 1. Mixed Precision (AMP) Setup
        # Only enable AMP if we are on a CUDA GPU
        self.use_amp = (self.device.type == 'cuda')
        self.scaler = torch.amp.GradScaler(enabled=self.use_amp)
        
        # 2. Global Confusion Matrix for mathematically correct mIoU
        # Shape: (num_classes, num_classes) -> initialized to zeros
   

        self.metrics = SegmentationMetrics(
            num_classes=self.num_classes,
            class_names=self.class_names,
            device=self.device
        )
        
        # Tracking variables for early stopping and best model
        self.best_miou = 0.0
        self.epochs_no_improve = 0
        self.start_epoch = 1

 

    def train_one_epoch(self, epoch: int) -> float:
        """Runs one full training epoch with Mixed Precision."""
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}", leave=False)
        
        for batch in pbar:
            # non_blocking=True speeds up CPU->GPU transfer when pin_memory=True
            images = batch["image"].to(self.device, non_blocking=True)
            masks = batch["mask"].to(self.device, non_blocking=True)
            
            # set_to_none=True is slightly faster than standard zero_grad()
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with Automatic Mixed Precision
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
            # Backward pass with Gradient Scaling
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Accumulate loss (weighted by batch size for accurate averaging)
            running_loss += loss.item() * images.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        return running_loss / len(self.train_loader.dataset)

    def validate(self, epoch: int) -> tuple[float, Dict]:
        """Runs validation, updates the global confusion matrix, and calculates metrics."""
        self.model.eval()
        running_loss = 0.0
        pbar = tqdm(self.val_loader, desc=f"Epoch {epoch}", leave=False)
        
        with torch.no_grad():
            for batch in pbar:
                images = batch["image"].to(self.device, non_blocking=True)
                masks = batch["mask"].to(self.device, non_blocking=True)
                
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                    
                # Get class predictions (argmax across the channel dimension)
                preds = torch.argmax(outputs, dim=1)
                
                # Update the global confusion matrix
                #self._update_confusion_matrix(preds, masks)
                self.metrics.update(preds, masks)
                
                running_loss += loss.item() * images.size(0)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
                
        val_loss = running_loss / len(self.val_loader.dataset)
       # miou, pixel_acc = self._calculate_metrics()
        metrics = self.metrics.compute()
        self.metrics.reset()
        
        return val_loss, metrics

    def save_checkpoint(self, epoch: int, metrics: dict, is_best: bool) -> None:
        """Saves the COMPLETE state of the training process for perfect resumability."""
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(), # AMP GradScaler state
            "best_miou": self.best_miou,
            "epochs_no_improve": self.epochs_no_improve,
            "metrics": metrics,
        }
        
        last_path = os.path.join(self.checkpoint_dir, "last_model.pth")
        torch.save(state, last_path)
        
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, "best_model.pth")
            torch.save(state, best_path)
            print(f"  -> Saved new best model with mIoU: {metrics['miou']:.4f}")

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Loads the full training state to resume exactly where it left off.
        Returns the epoch to start from.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Resume checkpoint not found at: {checkpoint_path}")
            
        print(f"Resuming training from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        self.best_miou = checkpoint["best_miou"]
        self.epochs_no_improve = checkpoint["epochs_no_improve"]
        
        self.start_epoch = checkpoint["epoch"] + 1
        print(f"Resuming from Epoch {self.start_epoch}. Best mIoU so far: {self.best_miou:.4f}")
        
        

    def fit(self) -> None:
        """The main orchestrator. Loops through epochs, trains, validates, and handles early stopping."""
        print(f"Starting training for {self.epochs} epochs...")
        print("-" * 75)
        
        for epoch in range(self.start_epoch, self.epochs + 1):
            start_time = time.time()
            
            # 1. Train and Validate
            train_loss = self.train_one_epoch(epoch)
            val_loss, metrics = self.validate(epoch)
            miou = metrics['miou']
            pixel_acc = metrics["pixel_acc"]
            macro_f1 = metrics['macro_f1']
            macro_recall = metrics['macro_recall']
            macro_precision = metrics['macro_precision']
            per_class_iou = metrics['per_class_iou']
            per_class_f1 = metrics['per_class_f1']
            # 2. Step the learning rate scheduler
            # Note: This assumes CosineAnnealingLR. If using ReduceLROnPlateau, 
            # you would pass the metric: self.scheduler.step(val_loss)
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # 3. Print Epoch Summary
            print(f"Epoch [{epoch:03d}/{self.epochs}] | "
                  f"Time: {epoch_time:5.1f}s | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"mIoU: {miou:.4f} | "
                  f"Pix Acc: {pixel_acc:.4f} | "
                  f"macro_f1: {macro_f1:.4f} |"
                  f"macro_recall: {macro_recall:.4f} |"
                  f"macro_precision: {macro_precision:.4f} |"
                  f"per_class_iou: {per_class_iou} |"
                  f"per_class_f1: {per_class_f1} "
                  )
            
            # 4. Check for best model and Early Stopping
            is_best = miou > self.best_miou
            if is_best:
                self.best_miou = miou
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                
            self.save_checkpoint(epoch, metrics, is_best)
            
            if self.epochs_no_improve >= self.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs.")
                break
                
        print("-" * 75)
        print(f"Training complete! Best mIoU achieved: {self.best_miou:.4f}")