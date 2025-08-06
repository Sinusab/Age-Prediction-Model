# 🎯 Age Prediction Model

A comprehensive and improved system for age prediction from facial images using modern Deep Learning techniques.

## ✨ Key Features

- **Advanced Architecture**: Uses ResNet blocks, Attention mechanisms, and BatchNormalization
- **Smart Data Management**: Data caching for faster execution
- **Optimized Training**: Early Stopping, Learning Rate Scheduling, Mixed Precision Training
- **Data Augmentation**: Various techniques for improved generalization
- **Complete Logging System**: Full tracking of training process
- **Easy Interface**: Multiple modes for inference

## 📁 Project Structure

```
├── config.py                 # Settings and hyperparameters
├── data_preprocessing.py     # Data preprocessing
├── dataset.py               # Dataset classes and DataLoaders
├── models.py                # Model architectures
├── train.py                 # Training script
├── inference.py             # Prediction and testing
├── utils.py                 # Helper functions
├── README.md                # Usage guide
├── requirements.txt         # Dependencies
└── notebooks/               # Jupyter notebooks (optional)
```

## 🛠️ Installation and Setup

### 1. Install Dependencies

```bash
pip install torch torchvision torchaudio
pip install pillow numpy pandas scikit-learn
pip install tqdm matplotlib
pip install psutil  # for system information display
pip install gdown   # for dataset downloading
```

Or install all at once:
```bash
pip install -r requirements.txt
```

### 2. Dataset Preparation

This project uses the UTKFace dataset. The dataset will be automatically downloaded if not present:

**Dataset Information:**
- **Source**: UTKFace Dataset (https://susanqq.github.io/UTKFace/)
- **License**: Please refer to the original UTKFace dataset license and terms of use
- **Description**: Large-scale face dataset with age, gender, and ethnicity annotations

The preprocessing script will automatically:
- Download the dataset files if not present
- Extract and process images
- Calculate channel statistics
- Cache everything in `./data_cache` directory

### 3. Data Preprocessing (Run Once)

```bash
python data_preprocessing.py
```

This step:
- Downloads dataset files if needed
- Extracts and processes images
- Resizes images appropriately
- Calculates mean and standard deviation for channels
- Saves everything in `./data_cache` path

## 🚀 Model Training

### Basic Training

```bash
python train.py
```

### Training with Custom Settings

```python
from train import Trainer
from config import Config

# Custom settings
custom_config = Config()
custom_config.LEARNING_RATE = 5e-4
custom_config.BATCH_SIZE = 64
custom_config.MAX_EPOCHS = 100

# Training
trainer = Trainer(
    model_type='single_task',
    backbone='improved_cnn',
    config_obj=custom_config
)

history = trainer.train()
```

## 🔮 Prediction

### Interactive Mode

```bash
python inference.py --model ./models/best_model.pth --interactive
```

### Single Image Prediction

```bash
python inference.py --model ./models/best_model.pth \
                   --image path/to/image.jpg \
                   --sex 1 --race 2
```

### Batch Processing

First create a JSON file:

```json
{
  "images": ["image1.jpg", "image2.jpg", "image3.jpg"],
  "sexes": [0, 1, 0],
  "races": [1, 2, 3]
}
```

Then run:

```bash
python inference.py --model ./models/best_model.pth --batch batch_data.json
```

## ⚙️ Configuration

All settings can be modified in `config.py`:

### Important Settings

```python
# Training settings
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 10

# Model settings
IMAGE_SIZE = 224
DROPOUT_RATE = 0.3
HIDDEN_DIMS = [512, 128]

# Data Augmentation
HORIZONTAL_FLIP_PROB = 0.5
ROTATION_DEGREES = 10
COLOR_JITTER_BRIGHTNESS = 0.2
```

## 📊 Training Monitoring

### Log Files

All training information is saved in `./results/training_YYYYMMDD_HHMMSS.log`.

### Training Charts

Loss, MAE, Learning Rate and other metrics charts are saved in `./results/training_history.png`.

### Checkpoints

- `./models/best_model.pth`: Best model based on validation loss
- `./models/latest_model.pth`: Latest model
- `./models/checkpoint_epoch_X.pth`: Checkpoint for each epoch

## 🏗️ Model Architecture

### Base Model (ImprovedCNN)

- **Residual Blocks**: For improved gradient flow
- **Attention Mechanisms**: Channel and Spatial attention
- **BatchNormalization**: Accelerated convergence
- **Dropout**: Prevents overfitting

### EfficientNet Model (Optional)

```python
from models import create_model

model = create_model('single_task', 'efficientnet', pretrained=True)
```

## 📈 Performance Improvement

### 1. Stronger Data Augmentation

```python
# In dataset.py
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    # ...
])
```

### 2. Learning Rate Scheduling

```python
# In train.py
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=config.MAX_EPOCHS
)
```

### 3. Mixed Precision Training

```python
# In config.py
MIXED_PRECISION = True  # GPU only
```

### 4. Hyperparameter Tuning

```python
# Experiment with different values
LEARNING_RATE = [1e-3, 5e-4, 1e-4]
BATCH_SIZE = [16, 32, 64]
DROPOUT_RATE = [0.2, 0.3, 0.5]
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory error**:
   ```python
   config.BATCH_SIZE = 16  # Reduce batch size
   config.MIXED_PRECISION = True  # Enable mixed precision
   ```

2. **Overfitting**:
   ```python
   config.DROPOUT_RATE = 0.5  # Increase dropout
   config.WEIGHT_DECAY = 1e-3  # Increase weight decay
   ```

3. **Slow convergence**:
   ```python
   config.LEARNING_RATE = 5e-3  # Increase learning rate
   # or use different learning rate scheduler
   ```

## 📋 Complete Example

```python
# 1. Preprocessing (once)
from data_preprocessing import DataPreprocessor
preprocessor = DataPreprocessor()
preprocessor.process_all()

# 2. Training
from train import Trainer
trainer = Trainer(model_type='single_task', backbone='improved_cnn')
history = trainer.train()

# 3. Prediction
from inference import AgePredictor
predictor = AgePredictor('./models/best_model.pth')
result = predictor.predict_single_image('test_image.jpg', sex=1, race=2)
print(f"Predicted age: {result['predicted_age']}")
```

## 🎯 Expected Results

With default settings, the model should achieve:

- **Training MAE**: < 8 years
- **Validation MAE**: < 10 years
- **Training Time**: 2-4 hours (with GPU)

## 🤝 Contributing

To improve the project:

1. Create new Issues
2. Submit Pull Requests
3. Share your suggestions

## 📄 License

This project is released under the MIT License.

## 📚 Dataset License and Attribution

This project uses the UTKFace dataset:
- **Dataset URL**: https://susanqq.github.io/UTKFace/
- **License**: Please refer to the original UTKFace dataset terms and conditions
- **Citation**: If you use this dataset, please cite the original UTKFace paper and follow their attribution requirements

---

**Note**: This project is designed for educational and research purposes. Commercial use requires additional considerations and compliance with dataset licensing terms.