"""
OCR Processor Module for AI Legal Document Explainer

This module handles Optical Character Recognition (OCR) processing for:
- Scanned documents
- PDF images
- Handwritten text
- Low-quality text images
"""

import logging
import cv2
import numpy as np
from PIL import Image
import pytesseract
from typing import Dict, List, Optional, Union, Tuple
from pathlib import Path
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OCRProcessor:
    """OCR processor for extracting text from images and scanned documents."""
    
    def __init__(self, tesseract_path: Optional[str] = None):
        """
        Initialize OCR processor.
        
        Args:
            tesseract_path: Path to tesseract executable (if not in PATH)
        """
        self.tesseract_path = tesseract_path
        
        # Configure tesseract path if provided
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        
        # OCR configuration
        self.default_config = '--oem 3 --psm 6'  # Default OCR Engine Mode and Page Segmentation Mode
        
        # Image preprocessing parameters
        self.preprocessing_params = {
            'blur_kernel': (5, 5),
            'threshold_block_size': 11,
            'threshold_c': 2,
            'morphology_kernel': np.ones((3, 3), np.uint8)
        }
    
    def process_image(self, image_path: Union[str, Path], 
                     preprocessing: bool = True,
                     ocr_config: Optional[str] = None) -> Dict[str, any]:
        """
        Process an image and extract text using OCR.
        
        Args:
            image_path: Path to the image file
            preprocessing: Whether to apply image preprocessing
            ocr_config: Custom OCR configuration string
            
        Returns:
            Dictionary containing OCR results
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        logger.info(f"Processing image with OCR: {image_path}")
        
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Convert BGR to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Apply preprocessing if requested
            if preprocessing:
                processed_image = self._preprocess_image(image_rgb)
            else:
                processed_image = image_rgb
            
            # Perform OCR
            ocr_config = ocr_config or self.default_config
            text = pytesseract.image_to_string(processed_image, config=ocr_config)
            
            # Get additional OCR data
            ocr_data = pytesseract.image_to_data(processed_image, config=ocr_config, output_type=pytesseract.Output.DICT)
            
            # Get bounding boxes for detected text
            bounding_boxes = self._extract_bounding_boxes(ocr_data)
            
            # Get confidence scores
            confidence_scores = self._extract_confidence_scores(ocr_data)
            
            result = {
                'text': text.strip(),
                'confidence': np.mean(confidence_scores) if confidence_scores else 0,
                'bounding_boxes': bounding_boxes,
                'word_count': len(text.split()),
                'character_count': len(text),
                'image_path': str(image_path),
                'preprocessing_applied': preprocessing,
                'ocr_config': ocr_config
            }
            
            logger.info(f"OCR completed for {image_path}. Extracted {result['word_count']} words.")
            return result
            
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            raise
    
    def process_pdf_images(self, pdf_images: List[Image.Image], 
                          preprocessing: bool = True) -> List[Dict[str, any]]:
        """
        Process multiple PDF images.
        
        Args:
            pdf_images: List of PIL Image objects from PDF
            preprocessing: Whether to apply image preprocessing
            
        Returns:
            List of OCR results for each image
        """
        results = []
        
        for i, image in enumerate(pdf_images):
            try:
                # Convert PIL image to OpenCV format
                image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                
                # Apply preprocessing if requested
                if preprocessing:
                    processed_image = self._preprocess_image(image_rgb)
                else:
                    processed_image = image_rgb
                
                # Perform OCR
                text = pytesseract.image_to_string(processed_image, config=self.default_config)
                ocr_data = pytesseract.image_to_data(processed_image, config=self.default_config, 
                                                   output_type=pytesseract.Output.DICT)
                
                # Get confidence scores
                confidence_scores = self._extract_confidence_scores(ocr_data)
                
                result = {
                    'page_number': i + 1,
                    'text': text.strip(),
                    'confidence': np.mean(confidence_scores) if confidence_scores else 0,
                    'word_count': len(text.split()),
                    'character_count': len(text),
                    'preprocessing_applied': preprocessing
                }
                
                results.append(result)
                logger.info(f"OCR completed for page {i+1}. Extracted {result['word_count']} words.")
                
            except Exception as e:
                logger.error(f"Error processing PDF image page {i+1}: {e}")
                results.append({
                    'page_number': i + 1,
                    'text': '',
                    'confidence': 0,
                    'word_count': 0,
                    'character_count': 0,
                    'error': str(e)
                })
        
        return results
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply image preprocessing to improve OCR accuracy.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(gray, self.preprocessing_params['blur_kernel'], 0)
            
            # Apply adaptive thresholding
            thresholded = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                self.preprocessing_params['threshold_block_size'],
                self.preprocessing_params['threshold_c']
            )
            
            # Apply morphological operations to clean up the image
            kernel = self.preprocessing_params['morphology_kernel']
            cleaned = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
            
            return cleaned
            
        except Exception as e:
            logger.warning(f"Image preprocessing failed: {e}. Returning original image.")
            return image
    
    def _extract_bounding_boxes(self, ocr_data: Dict) -> List[Dict[str, int]]:
        """Extract bounding boxes for detected text regions."""
        bounding_boxes = []
        
        try:
            n_boxes = len(ocr_data['text'])
            for i in range(n_boxes):
                if int(ocr_data['conf'][i]) > 0:  # Only include text with confidence > 0
                    box = {
                        'x': ocr_data['left'][i],
                        'y': ocr_data['top'][i],
                        'width': ocr_data['width'][i],
                        'height': ocr_data['height'][i],
                        'text': ocr_data['text'][i],
                        'confidence': int(ocr_data['conf'][i])
                    }
                    bounding_boxes.append(box)
        except Exception as e:
            logger.warning(f"Could not extract bounding boxes: {e}")
        
        return bounding_boxes
    
    def _extract_confidence_scores(self, ocr_data: Dict) -> List[int]:
        """Extract confidence scores for detected text."""
        try:
            return [int(conf) for conf in ocr_data['conf'] if int(conf) > 0]
        except Exception as e:
            logger.warning(f"Could not extract confidence scores: {e}")
            return []
    
    def get_ocr_languages(self) -> List[str]:
        """Get list of available OCR languages."""
        try:
            return pytesseract.get_languages()
        except Exception as e:
            logger.warning(f"Could not get OCR languages: {e}")
            return ['eng']  # Default to English
    
    def set_ocr_language(self, language: str):
        """Set OCR language for text recognition."""
        try:
            available_languages = self.get_ocr_languages()
            if language in available_languages:
                self.default_config = f'--oem 3 --psm 6 -l {language}'
                logger.info(f"OCR language set to: {language}")
            else:
                logger.warning(f"Language {language} not available. Available: {available_languages}")
        except Exception as e:
            logger.error(f"Error setting OCR language: {e}")
    
    def enhance_ocr_accuracy(self, image: np.ndarray, 
                           method: str = 'adaptive_threshold') -> np.ndarray:
        """
        Apply advanced image enhancement techniques.
        
        Args:
            image: Input image
            method: Enhancement method ('adaptive_threshold', 'otsu', 'clahe')
            
        Returns:
            Enhanced image
        """
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            if method == 'adaptive_threshold':
                enhanced = cv2.adaptiveThreshold(
                    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
                )
            elif method == 'otsu':
                _, enhanced = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif method == 'clahe':
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(gray)
            else:
                enhanced = gray
            
            return enhanced
            
        except Exception as e:
            logger.warning(f"Image enhancement failed: {e}. Returning original image.")
            return image


def main():
    """Test function for the OCR processor."""
    processor = OCRProcessor()
    
    # Test with a sample image if available
    test_images = [
        'sample.png',
        'sample.jpg',
        'sample.jpeg',
        'sample.tiff'
    ]
    
    for test_image in test_images:
        if Path(test_image).exists():
            try:
                result = processor.process_image(test_image, preprocessing=True)
                print(f"\nOCR Results for {test_image}:")
                print(f"Text: {result['text'][:100]}...")
                print(f"Confidence: {result['confidence']:.2f}")
                print(f"Word count: {result['word_count']}")
            except Exception as e:
                print(f"Error testing {test_image}: {e}")
        else:
            print(f"Test image {test_image} not found")
    
    # Test available languages
    languages = processor.get_ocr_languages()
    print(f"\nAvailable OCR languages: {languages}")


if __name__ == "__main__":
    main()
