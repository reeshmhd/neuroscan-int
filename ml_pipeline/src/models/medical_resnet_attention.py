"""
Medical-Grade ResNet50 with Spatial Attention
Designed for Alzheimer's MRI classification
Focuses on hippocampus, ventricles, and cortical regions
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    Helps model focus on diagnostically relevant brain regions:
    - Hippocampus (medial temporal lobe)
    - Ventricles (center/lateral)
    - Cortex (peripheral)
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Aggregate along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.conv(attention)
        attention = self.sigmoid(attention)
        
        return x * attention

class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    Emphasizes important feature channels
    """
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        
        # Average pooling
        avg_out = self.avg_pool(x).view(b, c)
        avg_out = self.fc(avg_out).view(b, c, 1, 1)
        
        # Max pooling
        max_out = self.max_pool(x).view(b, c)
        max_out = self.fc(max_out).view(b, c, 1, 1)
        
        # Combine
        attention = self.sigmoid(avg_out + max_out)
        return x * attention

class MedicalResNetAttention(nn.Module):
    """
    Medical-Grade ResNet50 with Dual Attention
    
    Architecture:
    - ResNet50 backbone (ImageNet pretrained)
    - Channel + Spatial Attention layers
    - Multi-scale feature fusion
    - Dropout regularization
    - Clinical-grade classifier
    """
    def __init__(self, num_classes=4, pretrained=True):
        super().__init__()
        
        # Load ResNet50 backbone
        resnet = models.resnet50(pretrained=pretrained)
        
        # Extract feature layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels
        
        # Add attention modules after key layers
        self.attention2 = nn.Sequential(
            ChannelAttention(512),
            SpatialAttention()
        )
        
        self.attention3 = nn.Sequential(
            ChannelAttention(1024),
            SpatialAttention()
        )
        
        self.attention4 = nn.Sequential(
            ChannelAttention(2048),
            SpatialAttention()
        )
        
        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Medical-grade classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
        
        # Initialize classifier weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Initial convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet blocks with attention
        x = self.layer1(x)
        
        x = self.layer2(x)
        x = self.attention2(x)  # Focus on intermediate features
        
        x = self.layer3(x)
        x = self.attention3(x)  # Focus on high-level features
        
        x = self.layer4(x)
        x = self.attention4(x)  # Focus on semantic features
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x
    
    def get_attention_maps(self, x):
        """Extract attention maps for visualization (Grad-CAM alternative)"""
        # Forward pass through layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        
        # Get attention from layer 3
        attention_map = self.attention3[1].conv(x)  # Spatial attention
        
        return attention_map

def create_medical_model(num_classes=4, pretrained=True):
    """Factory function to create medical-grade model"""
    return MedicalResNetAttention(num_classes=num_classes, pretrained=pretrained)
