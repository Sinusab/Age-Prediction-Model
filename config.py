import torch
import os
from dataclasses import dataclass
from typing import Tuple, List

@dataclass
class Config:
    """Configuration management and hyperparameters for the project"""
    
    # Paths
    EXTRACT_PATH: str = "./extracted_files"
    DATA_CACHE_PATH: str = "./data_cache"
    MODEL_SAVE_PATH: str = "./models"
    RESULTS_PATH: str = "./results"
    
    # Dataset settings
    TAR_FILES: List[str] = None
    IMAGE_SIZE: int = 224
    BATCH_SIZE: int = 32
    NUM_WORKERS: int = 0
    PIN_MEMORY: bool = True
    
    # Model settings
    NUM_CLASSES_SEX: int = 2
    NUM_CLASSES_RACE: int = 5
    DROPOUT_RATE: float = 0.3
    HIDDEN_DIMS: List[int] = None
    
    # Training settings
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    MAX_EPOCHS: int = 50
    EARLY_STOPPING_PATIENCE: int = 10
    EARLY_STOPPING_MIN_DELTA: float = 0.01
    LR_SCHEDULER_FACTOR: float = 0.5
    LR_SCHEDULER_PATIENCE: int = 5
    GRADIENT_CLIP_VALUE: float = 1.0
    
    # Data splitting settings
    TRAIN_SPLIT: float = 0.8
    VAL_SPLIT: float = 0.2
    RANDOM_STATE: int = 42
    
    # Data Augmentation settings
    HORIZONTAL_FLIP_PROB: float = 0.5
    ROTATION_DEGREES: int = 10
    COLOR_JITTER_BRIGHTNESS: float = 0.2
    COLOR_JITTER_CONTRAST: float = 0.2
    COLOR_JITTER_SATURATION: float = 0.2
    COLOR_JITTER_HUE: float = 0.1
    
    # GPU settings
    USE_CUDA: bool = True
    CUDA_BENCHMARK: bool = True
    MIXED_PRECISION: bool = True
    
    def __post_init__(self):
        """Settings after object creation"""
        if self.TAR_FILES is None:
            self.TAR_FILES = ['part1.tar.gz', 'part2.tar.gz', 'part3.tar.gz']
        
        if self.HIDDEN_DIMS is None:
            self.HIDDEN_DIMS = [512, 128]
        
        # Device configuration
        if self.USE_CUDA and torch.cuda.is_available():
            self.device = torch.device("cuda")
            if self.CUDA_BENCHMARK:
                torch.backends.cudnn.benchmark = True
        else:
            self.device = torch.device("cpu")
            self.MIXED_PRECISION = False  # GPU only
        
        # Create directories
        os.makedirs(self.DATA_CACHE_PATH, exist_ok=True)
        os.makedirs(self.MODEL_SAVE_PATH, exist_ok=True)
        os.makedirs(self.RESULTS_PATH, exist_ok=True)
    
    def get_cache_paths(self) -> dict:
        """Cache paths for storing processed data"""
        return {
            'images': os.path.join(self.DATA_CACHE_PATH, 'images.npy'),
            'ages': os.path.join(self.DATA_CACHE_PATH, 'ages.npy'),
            'sexes': os.path.join(self.DATA_CACHE_PATH, 'sexes.npy'),
            'races': os.path.join(self.DATA_CACHE_PATH, 'races.npy'),
            'stats': os.path.join(self.DATA_CACHE_PATH, 'dataset_stats.npz')
        }
    
    def print_config(self):
        """Print current configuration"""
        print("=" * 50)
        print("Project Configuration:")
        print("=" * 50)
        print(f"Device: {self.device}")
        print(f"Batch Size: {self.BATCH_SIZE}")
        print(f"Learning Rate: {self.LEARNING_RATE}")
        print(f"Max Epochs: {self.MAX_EPOCHS}")
        print(f"Image Size: {self.IMAGE_SIZE}")
        print(f"Mixed Precision: {self.MIXED_PRECISION}")
        print("=" * 50)

# Create default instance
config = Config()