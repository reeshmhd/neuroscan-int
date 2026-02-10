#!/usr/bin/env python3
"""
FIXED MRI-Only Models for Alzheimer's Detection - RGB Compatible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MRIOnlyAlzheimerNet(nn.Module):
    """Custom CNN for MRI-only Alzheimer's detection - FIXED for RGB input"""
    
    def __init__(self, num_classes=4, dropout_rate=0.5):
        super(MRIOnlyAlzheimerNet, self).__init__()
        
        # 🔥 CRITICAL FIX: Changed from 1 to 3 input channels for RGB
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)  # ✅ 3 channels for RGB
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Convolutional blocks
        self.conv_block1 = self._make_conv_block(32, 64)
        self.conv_block2 = self._make_conv_block(64, 128) 
        self.conv_block3 = self._make_conv_block(128, 256)
        self.conv_block4 = self._make_conv_block(256, 512)
        
        # Global Average Pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Classifier
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(512, num_classes)
        
        self._initialize_weights()
    
    def _make_conv_block(self, in_channels, out_channels):
        """Create a convolutional block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input: [batch_size, 3, 224, 224] ✅ Now supports RGB
        
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Convolutional blocks
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        x = self.conv_block4(x)
        
        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        features = x  # For feature extraction if needed
        
        x = self.dropout(x)
        logits = self.fc(x)
        
        return {
            'logits': logits,
            'features': features,
            'probabilities': F.softmax(logits, dim=1)
        }

class TransferLearningMRINet(nn.Module):
    """Transfer learning model using ResNet50 - FIXED for RGB input"""
    
    def __init__(self, num_classes=4, pretrained=True, dropout_rate=0.5):
        super(TransferLearningMRINet, self).__init__()
        
        # Load pretrained ResNet50 (already expects RGB input)
        self.backbone = models.resnet50(pretrained=pretrained)
        
        # Remove original classifier
        self.backbone.fc = nn.Identity()
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, num_classes)
        )
        
        # ✅ ResNet50 already expects 3-channel RGB input, so no changes needed
        
    def forward(self, x):
        # Input: [batch_size, 3, 224, 224] ✅ ResNet50 naturally supports RGB
        
        # Extract features using ResNet backbone
        features = self.backbone(x)
        
        # Classify
        logits = self.classifier(features)
        
        return {
            'logits': logits,
            'features': features,
            'probabilities': F.softmax(logits, dim=1)
        }

class BasicMRIClassifier(nn.Module):
    """Simple baseline model - FIXED for RGB input"""
    
    def __init__(self, num_classes=4):
        super(BasicMRIClassifier, self).__init__()
        
        # 🔥 CRITICAL FIX: Changed from 1 to 3 input channels for RGB
        self.features = nn.Sequential(
            # First conv block
            nn.Conv2d(3, 16, kernel_size=3, padding=1),  # ✅ 3 channels for RGB
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            # Second conv block
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(2),
            
            # Third conv block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Input: [batch_size, 3, 224, 224] ✅ Now supports RGB
        
        features = self.features(x)
        logits = self.classifier(features)
        
        return {
            'logits': logits,
            'features': features.view(features.size(0), -1),
            'probabilities': F.softmax(logits, dim=1)
        }

def create_model(model_type='custom', num_classes=4, **kwargs):
    """Factory function to create models"""
    
    if model_type == 'custom':
        return MRIOnlyAlzheimerNet(num_classes=num_classes, **kwargs)
    elif model_type == 'transfer':
        return TransferLearningMRINet(num_classes=num_classes, **kwargs)
    elif model_type == 'basic':
        return BasicMRIClassifier(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# Test function
def test_models():
    """Test all models with RGB input"""
    print("Testing RGB-compatible models...")
    
    batch_size = 2
    # ✅ RGB input: 3 channels
    dummy_input = torch.randn(batch_size, 3, 224, 224)
    
    models_to_test = ['custom', 'transfer', 'basic']
    
    for model_type in models_to_test:
        print(f"\n🧠 Testing {model_type} model...")
        
        try:
            model = create_model(model_type=model_type, num_classes=4)
            model.eval()
            
            with torch.no_grad():
                output = model(dummy_input)
            
            print(f"   ✅ Input shape: {dummy_input.shape}")
            print(f"   ✅ Output logits shape: {output['logits'].shape}")
            print(f"   ✅ Output probabilities shape: {output['probabilities'].shape}")
            print(f"   ✅ Model works with RGB input!")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")

if __name__ == "__main__":
    test_models()
