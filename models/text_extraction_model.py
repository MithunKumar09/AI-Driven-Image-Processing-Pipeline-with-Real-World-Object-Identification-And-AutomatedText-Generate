# models/text_extraction_model.py
from PIL import Image
import pytesseract
import cv2
import numpy as np
import re
import easyocr

class TextExtractionModel:
    def __init__(self):
        self.reader = easyocr.Reader(['en'])

    def extract_text(self, image):
        results = self.reader.readtext(image)
        # Extract just the text from the results
        text = ' '.join([result[1] for result in results])
        return text
    
    def clean_text(self, text):
        # Remove special characters and numbers
        cleaned_text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
        cleaned_text = re.sub(r'\d+', '', cleaned_text)  # Remove numbers
        
        # Replace multiple spaces with a single space
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        # Capitalize the first letter of each word for consistency
        cleaned_text = ' '.join(word.capitalize() for word in cleaned_text.split())

        return cleaned_text
