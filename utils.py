import os
import torch
import logging
import json
import numpy as np
from datetime import datetime
from typing import Dict, List, Any

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True, verbose=True):
        """
        Args:
            patience: Number of epochs to wait
            min_delta: Minimum change for improvement
            restore_best_weights: Whether to restore best model weights
            verbose: Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0

    def __call__(self, val_loss, model):
        """
        Args:
            val_loss: Current validation loss
            model: PyTorch model
            
        Returns:
            bool: Whether training should be stopped
        """
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
            if self.verbose:
                print(f"Validation loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience}")

        if self.counter >= self.patience:
            self.stopped_epoch = self.counter
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
                if self.verbose:
                    print("Restored best weights")
            return True
        
        return False

    def save_checkpoint(self, model):
        """Save best weights"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

class ModelCheckpoint:
    """Save model checkpoints"""
    
    def __init__(self, filepath, monitor='val_loss', save_best_only=True, 
                 mode='min', verbose=True):
        """
        Args:
            filepath: Path to save files
            monitor: Metric to monitor
            save_best_only: Save only best model
            mode: 'min' or 'max' for improvement
            verbose: Print messages
        """
        self.filepath = filepath
        self.monitor = monitor
        self.save_best_only = save_best_only
        self.mode = mode
        self.verbose = verbose
        
        if mode == 'min':
            self.best = np.inf
            self.monitor_op = np.less
        else:
            self.best = -np.inf
            self.monitor_op = np.greater
            
        os.makedirs(filepath, exist_ok=True)
    
    def save_checkpoint(self, state_dict, is_best=False):
        """
        Save checkpoint
        
        Args:
            state_dict: Dict containing model information
            is_best: Whether this is the best model
        """
        epoch = state_dict.get('epoch', 0)
        
        # Save current checkpoint
        checkpoint_path = os.path.join(self.filepath, f'checkpoint_epoch_{epoch}.pth')
        torch.save(state_dict, checkpoint_path)
        
        if self.verbose:
            print(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.filepath, 'best_model.pth')
            torch.save(state_dict, best_path)
            if self.verbose:
                print(f"Best model saved: {best_path}")
        
        # Save latest model
        latest_path = os.path.join(self.filepath, 'latest_model.pth')
        torch.save(state_dict, latest_path)

class MetricsTracker:
    """Track and save training metrics"""
    
    def __init__(self):
        self.metrics_history = []
        self.current_metrics = {}
    
    def update(self, metrics_dict: Dict[str, Any]):
        """Update metrics"""
        self.current_metrics.update(metrics_dict)
        self.metrics_history.append(self.current_metrics.copy())
    
    def get_current(self) -> Dict[str, Any]:
        """Get current metrics"""
        return self.current_metrics.copy()
    
    def get_history(self) -> List[Dict[str, Any]]:
        """Get complete metrics history"""
        return self.metrics_history.copy()
    
    def get_best(self, metric_name: str, mode='min') -> Dict[str, Any]:
        """Get best value of a metric"""
        if not self.metrics_history:
            return {}
        
        if mode == 'min':
            best_entry = min(self.metrics_history, key=lambda x: x.get(metric_name, float('inf')))
        else:
            best_entry = max(self.metrics_history, key=lambda x: x.get(metric_name, -float('inf')))
        
        return best_entry
    
    def save_to_json(self, filepath: str):
        """Save metrics to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.metrics_history, f, indent=2, ensure_ascii=False)
    
    def load_from_json(self, filepath: str):
        """Load metrics from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            self.metrics_history = json.load(f)
    
    def print_summary(self):
        """Print metrics summary"""
        if not self.metrics_history:
            print("No metrics recorded yet.")
            return
        
        print("\n" + "="*50)
        print("📊 Metrics Summary:")
        print("="*50)
        
        # Last epoch
        last_metrics = self.metrics_history[-1]
        print(f"Last Epoch: {last_metrics.get('epoch', 'N/A')}")
        print(f"Train Loss: {last_metrics.get('train_loss', 'N/A'):.4f}")
        print(f"Val Loss: {last_metrics.get('val_loss', 'N/A'):.4f}")
        print(f"Train MAE: {last_metrics.get('train_mae', 'N/A'):.4f}")
        print(f"Val MAE: {last_metrics.get('val_mae', 'N/A'):.4f}")
        
        # Best results
        best_val_loss = self.get_best('val_loss', 'min')
        best_val_mae = self.get_best('val_mae', 'min')
        
        print(f"\nBest Val Loss: {best_val_loss.get('val_loss', 'N/A'):.4f} (Epoch {best_val_loss.get('epoch', 'N/A')})")
        print(f"Best Val MAE: {best_val_mae.get('val_mae', 'N/A'):.4f} (Epoch {best_val_mae.get('epoch', 'N/A')})")
        print("="*50)

def setup_logging(log_dir: str, log_level=logging.INFO) -> logging.Logger:
    """Setup logging system"""
    
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')
    
    # Setup logger
    logger = logging.getLogger('AgePredictor')
    logger.setLevel(log_level)
    
    # Remove previous handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logging initialized. Log file: {log_file}")
    
    return logger

def load_checkpoint(checkpoint_path: str, model, optimizer=None, scheduler=None):
    """Load checkpoint"""
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer (optional)
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler (optional)
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Return additional information
    epoch = checkpoint.get('epoch', 0)
    val_loss = checkpoint.get('val_loss', None)
    val_mae = checkpoint.get('val_mae', None)
    
    print(f"Checkpoint loaded successfully:")
    print(f"  Epoch: {epoch}")
    print(f"  Val Loss: {val_loss}")
    print(f"  Val MAE: {val_mae}")
    
    return {
        'epoch': epoch,
        'val_loss': val_loss,
        'val_mae': val_mae,
        'mean_per_channel': checkpoint.get('mean_per_channel'),
        'std_per_channel': checkpoint.get('std_per_channel')
    }

def calculate_model_size(model):
    """Calculate model size"""
    param_size = 0
    param_count = 0
    
    for param in model.parameters():
        param_count += param.numel()
        param_size += param.numel() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    
    return {
        'param_count': param_count,
        'param_size_mb': param_size / 1024**2,
        'buffer_size_mb': buffer_size / 1024**2,
        'total_size_mb': size_all_mb
    }

def set_seed(seed=42):
    """Set seed for reproducibility"""
    import random
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def format_time(seconds):
    """Format time in readable format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        return f"{minutes}m {seconds}s"
    else:
        return f"{seconds}s"

def print_system_info():
    """Print system information"""
    print("="*50)
    print("🖥️ System Information:")
    print("="*50)
    
    # PyTorch version
    print(f"PyTorch Version: {torch.__version__}")
    
    # CUDA info
    if torch.cuda.is_available():
        print(f"CUDA Available: ✅")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"  GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        print("CUDA Available: ❌")
    
    # Memory info
    import psutil
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.total / 1024**3:.1f} GB")
    print(f"Available RAM: {memory.available / 1024**3:.1f} GB")
    
    print("="*50)

if __name__ == "__main__":
    # Test utility classes
    print("🧪 Testing Utility Classes...")
    
    # Test MetricsTracker
    tracker = MetricsTracker()
    tracker.update({'epoch': 1, 'train_loss': 0.5, 'val_loss': 0.6})
    tracker.update({'epoch': 2, 'train_loss': 0.4, 'val_loss': 0.5})
    tracker.print_summary()
    
    # Test logging
    logger = setup_logging('./test_logs')
    logger.info("Test log message")
    
    # Print system information
    print_system_info()
    
    print("✅ Utility classes test successful!")