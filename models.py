import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math
from config import config

class ResidualBlock(nn.Module):
    """Residual Block for improved gradient flow"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out

class ChannelAttention(nn.Module):
    """Channel attention mechanism for focusing on important features"""
    
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        
        # Max pooling  
        max_out = self.fc(self.max_pool(x).view(b, c))
        
        # Combine and apply attention
        out = avg_out + max_out
        attention = self.sigmoid(out).view(b, c, 1, 1)
        
        return x * attention.expand_as(x)

class SpatialAttention(nn.Module):
    """Spatial attention mechanism"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, 
                             padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(concat))
        
        return x * attention

class ImprovedCNN(nn.Module):
    """Improved CNN model with ResNet blocks and Attention mechanisms"""
    
    def __init__(self, in_channels=3):
        super(ImprovedCNN, self).__init__()
        
        # Initial convolution
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Attention modules
        self.ca1 = ChannelAttention(128)
        self.ca2 = ChannelAttention(256)
        self.ca3 = ChannelAttention(512)
        
        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.sa3 = SpatialAttention()
        
        # Global pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
        
        # Calculate output dimensions
        self.feature_size = 512
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        """Create layer with multiple residual blocks"""
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Initial convolution
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        # Residual layers with attention
        x = self.layer1(x)
        
        x = self.layer2(x)
        x = self.ca1(x)
        x = self.sa1(x)
        
        x = self.layer3(x)
        x = self.ca2(x)
        x = self.sa2(x)
        
        x = self.layer4(x)
        x = self.ca3(x)
        x = self.sa3(x)
        
        # Global pooling
        x = self.global_avg_pool(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        
        return x

class EfficientNetBackbone(nn.Module):
    """Use EfficientNet as backbone"""
    
    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super(EfficientNetBackbone, self).__init__()
        
        # Load model from torchvision (if available)
        try:
            if model_name == 'efficientnet_b0':
                self.backbone = models.efficientnet_b0(pretrained=pretrained)
                self.feature_size = 1280
            elif model_name == 'efficientnet_b1':
                self.backbone = models.efficientnet_b1(pretrained=pretrained)
                self.feature_size = 1280
            else:
                # Fallback to custom CNN
                self.backbone = ImprovedCNN()
                self.feature_size = 512
                return
                
            # Remove classifier layer
            self.backbone.classifier = nn.Identity()
            
        except:
            # If EfficientNet is not available, use custom model
            print("⚠️ EfficientNet not available, using custom model")
            self.backbone = ImprovedCNN()
            self.feature_size = 512
    
    def forward(self, x):
        return self.backbone(x)

class AgePredictor(nn.Module):
    """Final model for age prediction with improved architecture"""
    
    def __init__(self, backbone_type='improved_cnn', pretrained=True):
        super(AgePredictor, self).__init__()
        
        # Select backbone
        if backbone_type == 'efficientnet':
            self.backbone = EfficientNetBackbone(pretrained=pretrained)
        else:
            self.backbone = ImprovedCNN()
        
        # Calculate classifier input dimensions
        input_size = self.backbone.feature_size + config.NUM_CLASSES_SEX + config.NUM_CLASSES_RACE
        
        # Improved classifier
        self.classifier = nn.Sequential(
            nn.Linear(input_size, config.HIDDEN_DIMS[0]),
            nn.BatchNorm1d(config.HIDDEN_DIMS[0]),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            
            nn.Linear(config.HIDDEN_DIMS[0], config.HIDDEN_DIMS[1]),
            nn.BatchNorm1d(config.HIDDEN_DIMS[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE * 0.5),
            
            nn.Linear(config.HIDDEN_DIMS[1], 1)
        )
        
        # Weight initialization
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, image, sex, race):
        # Extract features from image
        img_features = self.backbone(image)
        
        # Combine features
        combined_features = torch.cat([img_features, sex, race], dim=1)
        
        # Predict age
        age_pred = self.classifier(combined_features)
        
        return age_pred

class MultiTaskAgePredictor(nn.Module):
    """Multi-task model that predicts age, gender, and race simultaneously"""
    
    def __init__(self, backbone_type='improved_cnn', pretrained=True):
        super(MultiTaskAgePredictor, self).__init__()
        
        # Shared backbone
        if backbone_type == 'efficientnet':
            self.backbone = EfficientNetBackbone(pretrained=pretrained)
        else:
            self.backbone = ImprovedCNN()
        
        feature_size = self.backbone.feature_size
        
        # Different branches for each task
        self.age_head = nn.Sequential(
            nn.Linear(feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        self.sex_head = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, config.NUM_CLASSES_SEX)
        )
        
        self.race_head = nn.Sequential(
            nn.Linear(feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, config.NUM_CLASSES_RACE)
        )
    
    def forward(self, image):
        features = self.backbone(image)
        
        age_pred = self.age_head(features)
        sex_pred = self.sex_head(features)
        race_pred = self.race_head(features)
        
        return age_pred, sex_pred, race_pred

def create_model(model_type='single_task', backbone='improved_cnn', pretrained=True):
    """Factory function for creating different models"""
    
    if model_type == 'single_task':
        return AgePredictor(backbone_type=backbone, pretrained=pretrained)
    elif model_type == 'multi_task':
        return MultiTaskAgePredictor(backbone_type=backbone, pretrained=pretrained)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

def count_parameters(model):
    """Count trainable parameters of the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def model_summary(model, input_size=(3, 224, 224)):
    """Model summary"""
    print("=" * 60)
    print("Model Summary:")
    print("=" * 60)
    print(f"Trainable parameters: {count_parameters(model):,}")
    
    # Calculate model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f"Model size: {size_all_mb:.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    # Test models
    print("🧪 Testing models...")
    
    # Single-task model
    model_single = create_model('single_task', 'improved_cnn')
    model_summary(model_single)
    
    # Test forward pass
    batch_size = 4
    dummy_img = torch.randn(batch_size, 3, 224, 224)
    dummy_sex = torch.randn(batch_size, 2)
    dummy_race = torch.randn(batch_size, 5)
    
    with torch.no_grad():
        output = model_single(dummy_img, dummy_sex, dummy_race)
        print(f"Single-task model output: {output.shape}")
    
    # Multi-task model
    model_multi = create_model('multi_task', 'improved_cnn')
    model_summary(model_multi)
    
    with torch.no_grad():
        age_out, sex_out, race_out = model_multi(dummy_img)
        print(f"Age output: {age_out.shape}")
        print(f"Gender output: {sex_out.shape}")
        print(f"Race output: {race_out.shape}")
    
    print("✅ Model testing successful!")