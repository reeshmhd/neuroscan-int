"""
Flask API for medical-grade Alzheimer's prediction
"""
from flask import Blueprint, request, jsonify
from backend.services.prediction_service import MedicalPredictionService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

prediction_bp = Blueprint('prediction', __name__)

# Initialize medical prediction service
try:
    predictor = MedicalPredictionService()
    logger.info("✅ Medical prediction service loaded successfully")
except Exception as e:
    logger.error(f"❌ Failed to load prediction service: {e}")
    predictor = None

@prediction_bp.route('/predict', methods=['POST'])
def predict():
    """
    POST /api/predict
    Medical-grade Alzheimer's prediction endpoint
    """
    if predictor is None:
        return jsonify({'error': 'Prediction service not available'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    try:
        image_bytes = image_file.read()
        result = predictor.predict(image_bytes)
        
        logger.info(f"Prediction: {result['predicted_class']} (confidence: {result['confidence']:.2f})")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500

@prediction_bp.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    status = {
        'status': 'healthy' if predictor is not None else 'unhealthy',
        'preprocessing': 'SimpleITK Medical-Grade',
        'model': 'MobileNetV2'
    }
    return jsonify(status), 200
