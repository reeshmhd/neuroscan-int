"""
Medical-grade preprocessing for JPG/JPEG MRI images
"""
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from skimage import exposure, filters
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MRIPreprocessor:
    """Preprocess JPG MRI images with medical-grade techniques"""
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
    
    def load_image(self, img_path):
        """Load JPG/JPEG image as grayscale"""
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.array(Image.open(img_path).convert('L'))
        return img
    
    def skull_strip_simple(self, img):
        """Simple skull stripping using thresholding"""
        # Otsu's thresholding
        thresh = filters.threshold_otsu(img)
        binary = img > thresh
        
        # Remove small objects
        from scipy import ndimage
        binary = ndimage.binary_fill_holes(binary)
        labeled, num = ndimage.label(binary)
        
        # Keep largest component (brain)
        if num > 0:
            sizes = ndimage.sum(binary, labeled, range(num + 1))
            mask = sizes < sizes.max()
            mask[0] = 0  # Keep background
            binary[mask[labeled]] = 0
        
        # Apply mask
        img_stripped = img.copy()
        img_stripped[~binary] = 0
        return img_stripped
    
    def intensity_normalization(self, img):
        """Z-score normalization within brain mask"""
        mask = img > 0
        if mask.sum() > 0:
            mean = img[mask].mean()
            std = img[mask].std()
            img_norm = np.zeros_like(img, dtype=np.float32)
            img_norm[mask] = (img[mask] - mean) / (std + 1e-8)
            return img_norm
        return img.astype(np.float32)
    
    def bias_field_correction_simple(self, img):
        """Simple bias field correction using histogram equalization"""
        # CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_corrected = clahe.apply(img)
        return img_corrected
    
    def preprocess(self, img_path, apply_skull_strip=True):
        """Complete preprocessing pipeline for JPG"""
        # 1. Load image
        img = self.load_image(img_path)
        
        # 2. Bias field correction
        img = self.bias_field_correction_simple(img)
        
        # 3. Skull stripping (optional)
        if apply_skull_strip:
            img = self.skull_strip_simple(img)
        
        # 4. Resize to target
        img = cv2.resize(img, self.target_size, interpolation=cv2.INTER_CUBIC)
        
        # 5. Intensity normalization
        img = self.intensity_normalization(img)
        
        return img
    
    def preprocess_dataset(self, input_dir, output_dir):
        """Preprocess entire dataset"""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        classes = ['Normal', 'VeryMild', 'Mild', 'Moderate']
        
        for cls in classes:
            cls_input = input_dir / cls
            cls_output = output_dir / cls
            cls_output.mkdir(parents=True, exist_ok=True)
            
            if not cls_input.exists():
                logger.warning(f"Class directory not found: {cls_input}")
                continue
            
            images = list(cls_input.glob('*.jpg')) + list(cls_input.glob('*.jpeg')) + list(cls_input.glob('*.png'))
            logger.info(f"Processing {len(images)} images for class {cls}...")
            
            for img_path in images:
                try:
                    # Preprocess
                    img_processed = self.preprocess(img_path, apply_skull_strip=True)
                    
                    # Save as .npy
                    output_path = cls_output / f"{img_path.stem}.npy"
                    np.save(output_path, img_processed)
                    
                except Exception as e:
                    logger.error(f"Error processing {img_path}: {e}")
            
            logger.info(f"✅ Completed {cls}: {len(list(cls_output.glob('*.npy')))} files")

def main():
    """Run preprocessing"""
    BASE = Path(r"C:\Users\user\OneDrive\Desktop\Project Australia\alzheimer_detection")
    INPUT = BASE / "data" / "raw" / "alzheimer_66k"
    OUTPUT = BASE / "data" / "preprocessed"
    
    preprocessor = MRIPreprocessor(target_size=(224, 224))
    preprocessor.preprocess_dataset(INPUT, OUTPUT)
    logger.info("✅ Preprocessing complete!")

if __name__ == "__main__":
    main()
