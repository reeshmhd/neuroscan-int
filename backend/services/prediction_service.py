"""
Prediction service using medical-grade model
"""
import torch
import numpy as np
from pathlib import Path
from torchvision import transforms
from backend.services.preprocessing_service import MedicalInferencePreprocessor
import logging

logger = logging.getLogger(__name__)

class MedicalPredictionService:
    """Production prediction service with medical-grade preprocessing"""
    
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models" / "saved_models" / "best_model_medical.pth"
        
        self.model = self._load_model(model_path)
        self.preprocessor = MedicalInferencePreprocessor()
        self.classes = ["Normal", "Very Mild Dementia", "Mild Dementia", "Moderate Dementia"]
        
        # Same transforms as training
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        logger.info(f"Medical prediction service initialized on {self.device}")
    
    def _load_model(self, model_path):
        """Load trained model"""
        from torchvision import models
        from torch import nn
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, 4)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def predict(self, image_bytes):
        """
        Run medical-grade prediction
        Returns: {class, confidence, probabilities, medical_notes}
        """
        try:
            # Medical preprocessing (SimpleITK pipeline)
            img_preprocessed = self.preprocessor.preprocess_uploaded_image(image_bytes)
            
            # Apply transforms
            img_tensor = self.transform(img_preprocessed).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(img_tensor)
                probabilities = torch.softmax(outputs, dim=1)[0]
                predicted_class = probabilities.argmax().item()
                confidence = probabilities[predicted_class].item()
            
            result = {
                'predicted_class': self.classes[predicted_class],
                'confidence': float(confidence),
                'probabilities': {
                    self.classes[i]: float(probabilities[i])
                    for i in range(len(self.classes))
                },
                'preprocessing': 'SimpleITK Medical-Grade',
                'medical_note': self._get_medical_note(predicted_class, confidence)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise
    
    def _get_medical_note(self, class_idx, confidence):
        """Generate medical interpretation note"""
        if confidence < 0.6:
            return "Low confidence prediction. Recommend clinical review."
        elif confidence < 0.8:
            return "Moderate confidence. Suggest additional diagnostic tests."
        else:
            return "High confidence prediction. Clinical correlation recommended."
