import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
from config import config
from data_preprocessing import DataPreprocessor

class ResizePad:
    """Custom transform for resize with aspect ratio preservation"""
    
    def __init__(self, size, fill=0):
        self.size = size
        self.fill = fill

    def __call__(self, img):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)

        w, h = img.size
        scale = self.size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img_resized = img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        pad_w = self.size - new_w
        pad_h = self.size - new_h
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top

        padding = (left, top, right, bottom)
        img_padded = TF.pad(img_resized, padding, fill=self.fill)
        return img_padded

class AgeDataset(Dataset):
    """Improved dataset for age prediction"""
    
    def __init__(self, images, ages, sexes, races, transform=None, is_training=True):
        """
        Args:
            images: numpy array of images
            ages: numpy array of ages
            sexes: numpy array of sex labels
            races: numpy array of race labels
            transform: torchvision transforms
            is_training: boolean indicating if this is training data
        """
        self.images = images
        self.ages = torch.tensor(ages, dtype=torch.float32)
        self.sexes = torch.tensor(sexes, dtype=torch.float32)
        self.races = torch.tensor(races, dtype=torch.float32)
        self.transform = transform
        self.is_training = is_training
        
        # If transform not specified, use default transform
        if self.transform is None:
            self.transform = self._get_default_transform()
    
    def _get_default_transform(self):
        """Default transform"""
        return transforms.Compose([
            transforms.ToPILImage(),
            ResizePad(config.IMAGE_SIZE, fill=0),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        image = self.images[idx]
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        # Load labels
        age = self.ages[idx]
        sex = self.sexes[idx]
        race = self.races[idx]
        
        return image, sex, race, age

class MultiTaskDataset(Dataset):
    """Dataset for multi-task training"""
    
    def __init__(self, images, ages, sexes, races, transform=None):
        self.images = images
        self.ages = torch.tensor(ages, dtype=torch.float32)
        self.sexes = torch.tensor(sexes, dtype=torch.long)  # For CrossEntropy
        self.races = torch.tensor(races, dtype=torch.long)  # For CrossEntropy
        self.transform = transform
        
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                ResizePad(config.IMAGE_SIZE, fill=0),
                transforms.ToTensor(),
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.transform:
            image = self.transform(image)
        
        age = self.ages[idx]
        sex = self.sexes[idx]
        race = self.races[idx]
        
        return image, age, sex, race

class TransformFactory:
    """Class for creating different transforms"""
    
    @staticmethod
    def get_train_transforms(mean_per_channel, std_per_channel, augment=True):
        """Transform for training data"""
        transforms_list = [
            transforms.ToPILImage(),
            ResizePad(config.IMAGE_SIZE, fill=0),
        ]
        
        if augment:
            # Data Augmentation
            transforms_list.extend([
                transforms.RandomHorizontalFlip(p=config.HORIZONTAL_FLIP_PROB),
                transforms.RandomRotation(config.ROTATION_DEGREES),
                transforms.ColorJitter(
                    brightness=config.COLOR_JITTER_BRIGHTNESS,
                    contrast=config.COLOR_JITTER_CONTRAST,
                    saturation=config.COLOR_JITTER_SATURATION,
                    hue=config.COLOR_JITTER_HUE
                ),
                transforms.RandomAffine(
                    degrees=0, 
                    translate=(0.1, 0.1), 
                    scale=(0.9, 1.1),
                    shear=5
                ),
            ])
        
        # Final transformation
        transforms_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean_per_channel.tolist(),
                std=std_per_channel.tolist()
            )
        ])
        
        return transforms.Compose(transforms_list)
    
    @staticmethod
    def get_val_transforms(mean_per_channel, std_per_channel):
        """Transform for validation data"""
        return transforms.Compose([
            transforms.ToPILImage(),
            ResizePad(config.IMAGE_SIZE, fill=0),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean_per_channel.tolist(),
                std=std_per_channel.tolist()
            )
        ])
    
    @staticmethod
    def get_test_transforms(mean_per_channel, std_per_channel):
        """Transform for test data"""
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=mean_per_channel.tolist(),
                std=std_per_channel.tolist()
            )
        ])

class DataLoaderFactory:
    """Class for creating different DataLoaders"""
    
    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.preprocessor = DataPreprocessor(self.config)
    
    def create_one_hot_encodings(self, sexes, races):
        """Create one-hot encoding"""
        sexes_onehot = np.eye(self.config.NUM_CLASSES_SEX)[sexes]
        races_onehot = np.eye(self.config.NUM_CLASSES_RACE)[races]
        return sexes_onehot, races_onehot
    
    def prepare_data(self, force_reprocess=False):
        """Prepare data"""
        # Load processed data
        data = self.preprocessor.process_all(force_reprocess=force_reprocess)
        images, ages, sexes, races, mean_per_channel, std_per_channel = data
        
        return images, ages, sexes, races, mean_per_channel, std_per_channel
    
    def create_single_task_loaders(self, force_reprocess=False, augment_train=True):
        """Create DataLoader for single-task training"""
        
        # Prepare data
        images, ages, sexes, races, mean_per_channel, std_per_channel = self.prepare_data(force_reprocess)
        
        # Create one-hot encodings
        sexes_onehot, races_onehot = self.create_one_hot_encodings(sexes, races)
        
        # Split data
        indices = np.arange(len(images))
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=self.config.VAL_SPLIT,
            random_state=self.config.RANDOM_STATE,
            stratify=sexes  # Balanced split based on gender
        )
        
        # Create transforms
        train_transform = TransformFactory.get_train_transforms(
            mean_per_channel, std_per_channel, augment=augment_train
        )
        val_transform = TransformFactory.get_val_transforms(
            mean_per_channel, std_per_channel
        )
        
        # Create datasets
        train_dataset = AgeDataset(
            images[train_idx], ages[train_idx], 
            sexes_onehot[train_idx], races_onehot[train_idx],
            transform=train_transform, is_training=True
        )
        
        val_dataset = AgeDataset(
            images[val_idx], ages[val_idx],
            sexes_onehot[val_idx], races_onehot[val_idx],
            transform=val_transform, is_training=False
        )
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )
        
        print(f"📊 Number of training samples: {len(train_dataset):,}")
        print(f"📊 Number of validation samples: {len(val_dataset):,}")
        print(f"📊 Number of training batches: {len(train_loader):,}")
        print(f"📊 Number of validation batches: {len(val_loader):,}")
        
        return train_loader, val_loader, mean_per_channel, std_per_channel
    
    def create_multi_task_loaders(self, force_reprocess=False):
        """Create DataLoader for multi-task training"""
        
        # Prepare data
        images, ages, sexes, races, mean_per_channel, std_per_channel = self.prepare_data(force_reprocess)
        
        # Split data
        indices = np.arange(len(images))
        train_idx, val_idx = train_test_split(
            indices, 
            test_size=self.config.VAL_SPLIT,
            random_state=self.config.RANDOM_STATE,
            stratify=sexes
        )
        
        # Create transforms
        train_transform = TransformFactory.get_train_transforms(
            mean_per_channel, std_per_channel, augment=True
        )
        val_transform = TransformFactory.get_val_transforms(
            mean_per_channel, std_per_channel
        )
        
        # Create datasets
        train_dataset = MultiTaskDataset(
            images[train_idx], ages[train_idx], 
            sexes[train_idx], races[train_idx],
            transform=train_transform
        )
        
        val_dataset = MultiTaskDataset(
            images[val_idx], ages[val_idx],
            sexes[val_idx], races[val_idx],
            transform=val_transform
        )
        
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=self.config.NUM_WORKERS,
            pin_memory=self.config.PIN_MEMORY
        )
        
        return train_loader, val_loader, mean_per_channel, std_per_channel

def test_dataset():
    """Test dataset and dataloader"""
    print("🧪 Testing Dataset and DataLoader...")
    
    # Create factory
    factory = DataLoaderFactory()
    
    try:
        # Test single-task
        train_loader, val_loader, mean, std = factory.create_single_task_loaders()
        
        # Test one batch
        for batch_idx, (images, sexes, races, ages) in enumerate(train_loader):
            print(f"Batch {batch_idx + 1}:")
            print(f"  Images shape: {images.shape}")
            print(f"  Sexes shape: {sexes.shape}")
            print(f"  Races shape: {races.shape}")
            print(f"  Ages shape: {ages.shape}")
            
            if batch_idx == 0:  # Test only one batch
                break
        
        print("✅ Dataset test successful!")
        
    except Exception as e:
        print(f"❌ Dataset test error: {str(e)}")

if __name__ == "__main__":
    test_dataset()