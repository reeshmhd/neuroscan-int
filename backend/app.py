#!/usr/bin/env python3
"""
BACKEND FOR REACT - COMPLETE FIX FOR app.py ONLY
Matches your api.ts and Results.tsx/Upload.tsx expectations
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
import io
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============ MODEL CLASS ============
class TransferLearningModel(nn.Module):
    def __init__(self, num_classes=4):
        super(TransferLearningModel, self).__init__()
        resnet = models.resnet50(pretrained=False)
        num_features = resnet.fc.in_features
        resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        self.model = resnet
    
    def forward(self, x):
        return self.model(x)

# ============ CONFIGURATION ============
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Robust path handling
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, '..', 'models', 'saved_models', 'resnet50_balanced_transfer.pth')
CLASS_NAMES = ['Normal', 'VeryMild', 'Mild', 'Moderate']

# Descriptions for each class
CLASS_DESCRIPTIONS = {
    'Normal': 'No signs of dementia detected. Maintain healthy lifestyle.',
    'VeryMild': 'Very mild cognitive impairment. Early monitoring recommended.',
    'Mild': 'Mild dementia detected. Consult with a healthcare provider.',
    'Moderate': 'Moderate dementia detected. Immediate specialist consultation required.'
}

# Severity levels
SEVERITY_LEVELS = {
    'Normal': 'low',
    'VeryMild': 'medium',
    'Mild': 'high',
    'Moderate': 'critical'
}

# Color codes for UI
CLASS_COLORS = {
    'Normal': '#10b981',      # Green
    'VeryMild': '#f59e0b',    # Amber
    'Mild': '#ef4444',        # Red
    'Moderate': '#7c3aed'     # Purple
}

# Load model
model = None
model_loaded = False

try:
    model = TransferLearningModel(num_classes=4).to(DEVICE)
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    model_loaded = True
    logger.info("✅ Model loaded successfully on {}".format(DEVICE))
except Exception as e:
    logger.error("❌ Failed to load model: {}".format(str(e)))
    model_loaded = False

# ============ ENDPOINTS ============

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'online',
        'model_loaded': model_loaded,
        'device': str(DEVICE)
    }), 200

@app.route('/api/predict', methods=['POST'])
def predict():
    """Main prediction endpoint - matches PredictionResult interface"""
    
    if not model_loaded:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 503
    
    if 'image' not in request.files:
        return jsonify({
            'success': False,
            'error': 'No image file provided'
        }), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({
            'success': False,
            'error': 'Empty filename'
        }), 400
    
    try:
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        logger.info("Processing image: {}".format(file.filename))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Preprocess image - EXACTLY same as training
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Apply transform
        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        image_tensor = image_tensor.to(DEVICE)
        
        # Get prediction
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class_idx = output.argmax(dim=1).item()
            predicted_class = CLASS_NAMES[predicted_class_idx]
            confidence = probabilities[0, predicted_class_idx].item()
        
        # Build response matching PredictionResult interface
        response = {
            'success': True,
            'predicted_class': predicted_class,
            'predicted_class_index': predicted_class_idx,
            'confidence': float(confidence),  # 0-1
            'confidence_percentage': float(confidence * 100),  # 0-100
            'description': CLASS_DESCRIPTIONS.get(predicted_class, 'Analysis complete.'),
            'probabilities': {
                CLASS_NAMES[i]: {
                    'probability': float(probabilities[0, i].item()),  # 0-1
                    'percentage': float(probabilities[0, i].item() * 100)  # 0-100
                }
                for i in range(len(CLASS_NAMES))
            },
            'severity_level': SEVERITY_LEVELS.get(predicted_class, 'low'),
            'color': CLASS_COLORS.get(predicted_class, '#3b82f6'),
            'model_info': {
                'architecture': 'ResNet50 Transfer Learning',
                'parameters': '23.5M',
                'device': str(DEVICE)
            }
        }
        
        logger.info("✅ Prediction: {} ({:.2f}%)".format(
            response['predicted_class'],
            response['confidence_percentage']
        ))
        
        return jsonify(response), 200
    
    except Exception as e:
        logger.error("❌ Error: {}".format(str(e)), exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ============ RUN SERVER ============
if __name__ == '__main__':
    logger.info("Starting Flask server on 0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
