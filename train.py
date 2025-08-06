import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from datetime import datetime

from config import config
from models import create_model, model_summary
from dataset import DataLoaderFactory
from utils import EarlyStopping, ModelCheckpoint, MetricsTracker, setup_logging

class Trainer:
    """Main class for model training"""
    
    def __init__(self, model_type='single_task', backbone='improved_cnn', 
                 pretrained=True, config_obj=None):
        
        self.config = config_obj or config
        self.device = self.config.device
        
        # Setup logging
        self.logger = setup_logging(self.config.RESULTS_PATH)
        
        # Create model
        self.model = create_model(model_type, backbone, pretrained)
        self.model.to(self.device)
        
        # Display model summary
        model_summary(self.model)
        
        # Setup loss function
        self.criterion = nn.MSELoss()
        
        # Setup optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        # Setup scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.LR_SCHEDULER_FACTOR,
            patience=self.config.LR_SCHEDULER_PATIENCE,
            verbose=True
        )
        
        # Mixed precision training
        self.scaler = GradScaler() if self.config.MIXED_PRECISION else None
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=self.config.EARLY_STOPPING_PATIENCE,
            min_delta=self.config.EARLY_STOPPING_MIN_DELTA,
            restore_best_weights=True
        )
        
        # Model checkpoint
        self.checkpoint = ModelCheckpoint(
            self.config.MODEL_SAVE_PATH,
            monitor='val_loss',
            save_best_only=True,
            verbose=True
        )
        
        # Metrics tracker
        self.metrics = MetricsTracker()
        
        # Data loaders
        self.data_factory = DataLoaderFactory(self.config)
        
        self.logger.info("Trainer initialized successfully")
    
    def train_epoch(self, train_loader, epoch):
        """Train one epoch"""
        self.model.train()
        running_loss = 0.0
        running_mae = 0.0
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, (images, sexes, races, ages) in enumerate(pbar):
            # Transfer data to GPU
            images = images.to(self.device, non_blocking=True)
            sexes = sexes.to(self.device, non_blocking=True)
            races = races.to(self.device, non_blocking=True)
            ages = ages.to(self.device, non_blocking=True).unsqueeze(1)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.config.MIXED_PRECISION:
                with autocast():
                    outputs = self.model(images, sexes, races)
                    loss = self.criterion(outputs, ages)
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.GRADIENT_CLIP_VALUE
                )
                
                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images, sexes, races)
                loss = self.criterion(outputs, ages)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.GRADIENT_CLIP_VALUE
                )
                
                self.optimizer.step()
            
            # Calculate metrics
            running_loss += loss.item()
            mae = torch.mean(torch.abs(outputs - ages)).item()
            running_mae += mae
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'MAE': f'{mae:.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        epoch_loss = running_loss / num_batches
        epoch_mae = running_mae / num_batches
        
        return epoch_loss, epoch_mae
    
    def validate_epoch(self, val_loader):
        """Validate one epoch"""
        self.model.eval()
        running_loss = 0.0
        running_mae = 0.0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for images, sexes, races, ages in tqdm(val_loader, desc="Validation"):
                images = images.to(self.device, non_blocking=True)
                sexes = sexes.to(self.device, non_blocking=True)
                races = races.to(self.device, non_blocking=True)
                ages = ages.to(self.device, non_blocking=True).unsqueeze(1)
                
                if self.config.MIXED_PRECISION:
                    with autocast():
                        outputs = self.model(images, sexes, races)
                        loss = self.criterion(outputs, ages)
                else:
                    outputs = self.model(images, sexes, races)
                    loss = self.criterion(outputs, ages)
                
                running_loss += loss.item()
                running_mae += torch.mean(torch.abs(outputs - ages)).item()
        
        epoch_loss = running_loss / num_batches
        epoch_mae = running_mae / num_batches
        
        return epoch_loss, epoch_mae
    
    def train(self, force_reprocess=False, augment_train=True):
        """Complete model training"""
        self.logger.info("Starting training...")
        self.config.print_config()
        
        # Prepare data
        train_loader, val_loader, mean_per_channel, std_per_channel = \
            self.data_factory.create_single_task_loaders(
                force_reprocess=force_reprocess,
                augment_train=augment_train
            )
        
        # Save statistics for use in inference
        self.mean_per_channel = mean_per_channel
        self.std_per_channel = std_per_channel
        
        start_time = time.time()
        best_val_loss = float('inf')
        
        for epoch in range(1, self.config.MAX_EPOCHS + 1):
            self.logger.info(f"\n=== Epoch {epoch}/{self.config.MAX_EPOCHS} ===")
            
            # Training
            train_loss, train_mae = self.train_epoch(train_loader, epoch)
            
            # Validation
            val_loss, val_mae = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Record metrics
            self.metrics.update({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_mae': train_mae,
                'val_loss': val_loss,
                'val_mae': val_mae,
                'lr': self.optimizer.param_groups[0]['lr']
            })
            
            # Log results
            self.logger.info(f"Train Loss: {train_loss:.4f}, Train MAE: {train_mae:.4f}")
            self.logger.info(f"Val Loss: {val_loss:.4f}, Val MAE: {val_mae:.4f}")
            self.logger.info(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                
            self.checkpoint.save_checkpoint({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'train_loss': train_loss,
                'train_mae': train_mae,
                'mean_per_channel': mean_per_channel,
                'std_per_channel': std_per_channel,
                'config': self.config
            }, is_best)
            
            # Early stopping
            if self.early_stopping(val_loss, self.model):
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        # Calculate total training time
        total_time = time.time() - start_time
        self.logger.info(f"\nTraining completed in {total_time/3600:.2f} hours")
        self.logger.info(f"Best validation loss: {best_val_loss:.4f}")
        
        # Plot training history
        self.plot_training_history()
        
        return self.metrics.get_history()
    
    def plot_training_history(self):
        """Plot training history"""
        history = self.metrics.get_history()
        
        if not history:
            return
        
        epochs = [h['epoch'] for h in history]
        train_losses = [h['train_loss'] for h in history]
        val_losses = [h['val_loss'] for h in history]
        train_maes = [h['train_mae'] for h in history]
        val_maes = [h['val_mae'] for h in history]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        ax1.plot(epochs, train_losses, label='Train Loss', color='blue')
        ax1.plot(epochs, val_losses, label='Val Loss', color='red')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # MAE plot
        ax2.plot(epochs, train_maes, label='Train MAE', color='blue')
        ax2.plot(epochs, val_maes, label='Val MAE', color='red')
        ax2.set_title('Training and Validation MAE')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MAE')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        lrs = [h['lr'] for h in history]
        ax3.plot(epochs, lrs, color='green')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)
        
        # Loss difference plot
        loss_diff = [abs(t - v) for t, v in zip(train_losses, val_losses)]
        ax4.plot(epochs, loss_diff, color='purple')
        ax4.set_title('Train-Val Loss Difference')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('|Train Loss - Val Loss|')
        ax4.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.config.RESULTS_PATH, 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self.logger.info(f"Training plots saved to {plot_path}")

def main():
    """Main function for running training"""
    print("🚀 Starting age prediction model training...")
    
    try:
        # Create trainer
        trainer = Trainer(
            model_type='single_task',
            backbone='improved_cnn',
            pretrained=True
        )
        
        # Train model
        history = trainer.train(
            force_reprocess=False,  # Use cache
            augment_train=True      # Enable data augmentation
        )
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()