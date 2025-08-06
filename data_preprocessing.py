import os
import tarfile
import re
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pickle
from config import config

# Import gdown for dataset downloading
try:
    import gdown
    GDOWN_AVAILABLE = True
except ImportError:
    GDOWN_AVAILABLE = False
    print("Warning: gdown not available. Please install it with 'pip install gdown' for automatic dataset downloading.")

class DataPreprocessor:
    """Class for data preprocessing and statistics computation"""
    
    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.cache_paths = self.config.get_cache_paths()
        
        # UTKFace dataset file IDs for Google Drive download
        self.file_ids = [
            '1mb5Z24TsnKI3ygNIlX6ZFiwUj0_PmpAW',
            '19vdaXVRtkP-nyxz1MYwXiFsh_m_OL72b',
            '1oj9ZWsLV2-k2idoW_nRSrLQLUP3hus3b'
        ]
        
        self.file_names = ['file1.tar.gz', 'file2.tar.gz', 'file3.tar.gz']
        
    def download_dataset(self):
        """Download UTKFace dataset files from Google Drive if not present"""
        if not GDOWN_AVAILABLE:
            print("❌ gdown is not available. Please install it with 'pip install gdown'")
            return False
            
        print("🌐 Checking for dataset files...")
        
        # Check if any tar files are missing
        missing_files = []
        for file_name in self.file_names:
            if not os.path.exists(file_name):
                missing_files.append(file_name)
        
        if not missing_files:
            print("✅ All dataset files are already present")
            return True
        
        print(f"📥 Downloading {len(missing_files)} missing dataset file(s)...")
        print("📚 Dataset: UTKFace (https://susanqq.github.io/UTKFace/)")
        print("⚖️  Please ensure you comply with the original dataset license and terms of use")
        
        for file_id, file_name in zip(self.file_ids, self.file_names):
            if file_name in missing_files:
                try:
                    url = f'https://drive.google.com/uc?id={file_id}'
                    print(f"📥 Downloading {file_name}...")
                    gdown.download(url, file_name, quiet=False)
                    print(f"✅ Downloaded {file_name}")
                except Exception as e:
                    print(f"❌ Failed to download {file_name}: {str(e)}")
                    return False
        
        print("✅ Dataset download completed!")
        return True
        
    def extract_tar_files(self):
        """Extract tar.gz files"""
        print("🗂️ Extracting tar.gz files...")
        
        if os.path.exists(self.config.EXTRACT_PATH):
            print(f"✅ Path {self.config.EXTRACT_PATH} already exists")
            return True
            
        os.makedirs(self.config.EXTRACT_PATH, exist_ok=True)
        
        for file_name in self.file_names:
            if os.path.exists(file_name):
                print(f"📦 Extracting {file_name}...")
                with tarfile.open(file_name, "r:gz") as tar:
                    tar.extractall(path=self.config.EXTRACT_PATH)
            else:
                print(f"⚠️ File {file_name} not found")
                
        return os.path.exists(self.config.EXTRACT_PATH)
    
    def extract_data_from_filename(self, filename):
        """Safe extraction of features from filename"""
        try:
            numbers = re.findall(r'\d+', filename)
            if len(numbers) < 3:
                return None
            
            age = int(numbers[0])
            sex = int(numbers[1])
            race = int(numbers[2])
            
            # Check value ranges
            if not (0 <= age <= 120):  # reasonable age
                return None
            if sex not in [0, 1]:  # gender
                return None
            if race not in [0, 1, 2, 3, 4]:  # race
                return None
                
            return age, sex, race
        except Exception as e:
            return None
    
    def load_and_process_images(self):
        """Load and process images"""
        print("🖼️ Loading and processing images...")
        
        # Find all image files
        image_files = []
        for root, dirs, files in os.walk(self.config.EXTRACT_PATH):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_files.append(os.path.join(root, file))
        
        print(f"📊 Number of files found: {len(image_files)}")
        
        images = []
        ages = []
        sexes = []
        races = []
        
        failed_count = 0
        
        for img_path in tqdm(image_files, desc="Processing images"):
            # Extract features from filename
            result = self.extract_data_from_filename(os.path.basename(img_path))
            if result is None:
                failed_count += 1
                continue
            
            try:
                age, sex, race = result
                
                # Load and resize image
                img = Image.open(img_path).convert('RGB')
                img = img.resize((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), Image.Resampling.LANCZOS)
                img_array = np.array(img, dtype=np.uint8)
                
                images.append(img_array)
                ages.append(age)
                sexes.append(sex)
                races.append(race)
                
            except Exception as e:
                failed_count += 1
                continue
        
        print(f"✅ Images processed: {len(images)}")
        print(f"❌ Failed images: {failed_count}")
        
        if len(images) == 0:
            raise ValueError("No valid images found!")
        
        # Convert to numpy arrays
        images = np.array(images)
        ages = np.array(ages)
        sexes = np.array(sexes)
        races = np.array(races)
        
        return images, ages, sexes, races
    
    def compute_dataset_statistics(self, images):
        """Compute dataset statistics (mean and standard deviation)"""
        print("📈 Computing dataset statistics...")
        
        N, H, W, C = images.shape
        sum_pixels = np.zeros(C, dtype=np.float64)
        sum_squared = np.zeros(C, dtype=np.float64)
        total_pixels = 0
        
        batch_size = 100
        for i in tqdm(range(0, N, batch_size), desc="Computing statistics"):
            batch = images[i:i+batch_size].astype(np.float64) / 255.0
            
            sum_pixels += batch.sum(axis=(0, 1, 2))
            sum_squared += (batch ** 2).sum(axis=(0, 1, 2))
            total_pixels += batch.shape[0] * H * W
        
        mean = sum_pixels / total_pixels
        variance = (sum_squared / total_pixels) - (mean ** 2)
        std = np.sqrt(variance)
        
        return mean, std
    
    def create_one_hot_encodings(self, sexes, races):
        """Create one-hot encoding for categorical features"""
        print("🔢 Creating one-hot encodings...")
        
        # One-hot encoding for gender
        sexes_onehot = np.eye(self.config.NUM_CLASSES_SEX)[sexes]
        
        # One-hot encoding for race
        races_onehot = np.eye(self.config.NUM_CLASSES_RACE)[races]
        
        return sexes_onehot, races_onehot
    
    def save_processed_data(self, images, ages, sexes, races, mean_per_channel, std_per_channel):
        """Save processed data"""
        print("💾 Saving processed data...")
        
        # Save main arrays
        np.save(self.cache_paths['images'], images)
        np.save(self.cache_paths['ages'], ages)
        np.save(self.cache_paths['sexes'], sexes)
        np.save(self.cache_paths['races'], races)
        
        # Save statistics
        np.savez(self.cache_paths['stats'], 
                mean=mean_per_channel, 
                std=std_per_channel)
        
        print(f"✅ Data saved in {self.config.DATA_CACHE_PATH}")
    
    def load_processed_data(self):
        """Load processed data from cache"""
        print("📂 Loading cached data...")
        
        try:
            images = np.load(self.cache_paths['images'])
            ages = np.load(self.cache_paths['ages'])
            sexes = np.load(self.cache_paths['sexes'])
            races = np.load(self.cache_paths['races'])
            
            stats = np.load(self.cache_paths['stats'])
            mean_per_channel = stats['mean']
            std_per_channel = stats['std']
            
            print("✅ Data loaded from cache")
            return images, ages, sexes, races, mean_per_channel, std_per_channel
            
        except FileNotFoundError:
            print("❌ Cache files not found")
            return None
    
    def is_cache_available(self):
        """Check cache availability"""
        required_files = [
            self.cache_paths['images'],
            self.cache_paths['ages'],
            self.cache_paths['sexes'],
            self.cache_paths['races'],
            self.cache_paths['stats']
        ]
        
        return all(os.path.exists(f) for f in required_files)
    
    def print_dataset_info(self, images, ages, sexes, races):
        """Print dataset information"""
        print("\n" + "="*50)
        print("📊 Dataset Information:")
        print("="*50)
        print(f"Number of images: {len(images):,}")
        print(f"Image dimensions: {images.shape[1:]}")
        print(f"Age range: {ages.min()} - {ages.max()}")
        print(f"Mean age: {ages.mean():.1f}")
        print(f"Age standard deviation: {ages.std():.1f}")
        
        print(f"\nGender distribution:")
        unique_sexes, counts_sexes = np.unique(sexes, return_counts=True)
        for sex, count in zip(unique_sexes, counts_sexes):
            gender_label = "Female" if sex == 0 else "Male"
            print(f"  {gender_label} ({sex}): {count:,} ({count/len(sexes)*100:.1f}%)")
        
        print(f"\nRace distribution:")
        unique_races, counts_races = np.unique(races, return_counts=True)
        race_labels = ["White", "Black", "Asian", "Indian", "Others"]
        for race, count in zip(unique_races, counts_races):
            race_label = race_labels[race] if race < len(race_labels) else f"Race {race}"
            print(f"  {race_label} ({race}): {count:,} ({count/len(races)*100:.1f}%)")
        
        print("="*50)
    
    def process_all(self, force_reprocess=False):
        """Complete data processing"""
        # Check cache
        if not force_reprocess and self.is_cache_available():
            print("📦 Using cached data...")
            return self.load_processed_data()
        
        print("🔄 Starting new data processing...")
        
        # Download dataset if needed
        if not self.download_dataset():
            print("❌ Failed to download dataset. Please download manually.")
            print("📚 Dataset URL: https://susanqq.github.io/UTKFace/")
        
        # Extract files
        if not self.extract_tar_files():
            raise FileNotFoundError("tar.gz files not found. Please ensure dataset files are available.")
        
        # Load and process images
        images, ages, sexes, races = self.load_and_process_images()
        
        # Print dataset information
        self.print_dataset_info(images, ages, sexes, races)
        
        # Compute statistics
        mean_per_channel, std_per_channel = self.compute_dataset_statistics(images)
        
        print(f"📊 Channel statistics:")
        print(f"Mean: {mean_per_channel}")
        print(f"Std: {std_per_channel}")
        
        # Save data
        self.save_processed_data(images, ages, sexes, races, mean_per_channel, std_per_channel)
        
        return images, ages, sexes, races, mean_per_channel, std_per_channel

def main():
    """Main function for running preprocessing"""
    print("🚀 Starting data preprocessing...")
    print("📚 Dataset: UTKFace (https://susanqq.github.io/UTKFace/)")
    print("⚖️  Please ensure compliance with the original dataset license and terms of use")
    
    preprocessor = DataPreprocessor()
    
    try:
        # Process data
        images, ages, sexes, races, mean_per_channel, std_per_channel = preprocessor.process_all()
        
        print("\n✅ Preprocessing completed successfully!")
        print(f"📁 Processed files saved in {config.DATA_CACHE_PATH}")
        print("📚 Dataset source: UTKFace (https://susanqq.github.io/UTKFace/)")
        print("⚖️  Remember to cite the original UTKFace dataset in your work")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in preprocessing: {str(e)}")
        print("💡 If download fails, please manually download the UTKFace dataset")
        print("📚 Dataset URL: https://susanqq.github.io/UTKFace/")
        return False

if __name__ == "__main__":
    success = main()