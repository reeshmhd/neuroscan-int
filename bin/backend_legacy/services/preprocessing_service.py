"""
Medical-grade preprocessing service for production inference
Uses SimpleITK to match training pipeline
"""
import SimpleITK as sitk
import numpy as np
from PIL import Image
import io
import logging

logger = logging.getLogger(__name__)

class MedicalInferencePreprocessor:
    """
    Production inference preprocessing using SimpleITK
    MUST match training preprocessing pipeline
    """
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        logger.info("Medical-grade inference preprocessor initialized (SimpleITK)")
    
    def preprocess_uploaded_image(self, image_bytes):
        """
        Preprocess uploaded image for inference
        Matches SimpleITK training pipeline
        """
        try:
            # Load from bytes
            pil_img = Image.open(io.BytesIO(image_bytes)).convert('L')
            arr = np.array(pil_img, dtype=np.float32)
            
            # Convert to SimpleITK
            sitk_img = sitk.GetImageFromArray(arr)
            sitk_img.SetSpacing([1.0, 1.0])
            
            # N4 Bias Correction (same as training)
            sitk_img = self._n4_bias_correction(sitk_img)
            
            # Skull stripping (same as training)
            sitk_img = self._skull_strip(sitk_img)
            
            # Resample (same as training)
            sitk_img = self._resample(sitk_img, self.target_size)
            
            # Intensity normalization (same as training)
            sitk_img = self._normalize(sitk_img)
            
            # Convert to numpy
            arr_processed = sitk.GetArrayFromImage(sitk_img)
            
            # Convert to 3-channel
            img_3ch = np.stack([arr_processed, arr_processed, arr_processed], axis=2)
            
            # Normalize to [0, 255]
            img_min, img_max = img_3ch.min(), img_3ch.max()
            if img_max > img_min:
                img_3ch = ((img_3ch - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img_3ch = np.zeros_like(img_3ch, dtype=np.uint8)
            
            return img_3ch
            
        except Exception as e:
            logger.error(f"Preprocessing failed: {e}")
            raise
    
    def _n4_bias_correction(self, image):
        """N4 bias correction (matches training)"""
        try:
            if image.GetPixelID() != sitk.sitkFloat32:
                image = sitk.Cast(image, sitk.sitkFloat32)
            
            mask = sitk.OtsuThreshold(image, 0, 1)
            corrector = sitk.N4BiasFieldCorrectionImageFilter()
            corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])
            return corrector.Execute(image, mask)
        except:
            return image
    
    def _skull_strip(self, image):
        """Skull stripping (matches training)"""
        try:
            otsu = sitk.OtsuThresholdImageFilter()
            otsu.SetInsideValue(0)
            otsu.SetOutsideValue(1)
            mask = otsu.Execute(image)
            
            mask = sitk.BinaryMorphologicalClosing(mask, [3, 3])
            mask = sitk.BinaryFillhole(mask)
            
            cc = sitk.ConnectedComponent(mask)
            stats = sitk.LabelShapeStatisticsImageFilter()
            stats.Execute(cc)
            
            if stats.GetNumberOfLabels() > 0:
                labels = stats.GetLabels()
                largest = max(labels, key=lambda l: stats.GetPhysicalSize(l))
                mask = sitk.Equal(cc, largest)
            
            return sitk.Mask(image, mask)
        except:
            return image
    
    def _resample(self, image, new_size):
        """Resample (matches training)"""
        try:
            original_size = image.GetSize()
            original_spacing = image.GetSpacing()
            
            new_spacing = [
                (original_size[i] * original_spacing[i]) / new_size[i]
                for i in range(2)
            ]
            
            resampler = sitk.ResampleImageFilter()
            resampler.SetSize(new_size)
            resampler.SetOutputSpacing(new_spacing)
            resampler.SetOutputOrigin(image.GetOrigin())
            resampler.SetOutputDirection(image.GetDirection())
            resampler.SetInterpolator(sitk.sitkBSpline)
            resampler.SetDefaultPixelValue(0)
            resampler.SetTransform(sitk.Transform())
            
            return resampler.Execute(image)
        except:
            return image
    
    def _normalize(self, image):
        """Z-score normalization (matches training)"""
        try:
            arr = sitk.GetArrayFromImage(image)
            mask = arr > 0
            
            if mask.sum() > 0:
                mean, std = arr[mask].mean(), arr[mask].std()
                arr_norm = np.zeros_like(arr, dtype=np.float32)
                arr_norm[mask] = (arr[mask] - mean) / (std + 1e-8)
            else:
                arr_norm = arr.astype(np.float32)
            
            img_norm = sitk.GetImageFromArray(arr_norm)
            img_norm.CopyInformation(image)
            return img_norm
        except:
            return image
