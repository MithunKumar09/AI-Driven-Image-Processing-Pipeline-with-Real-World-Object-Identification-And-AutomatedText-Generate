# tests/test_text_extraction.py
from PIL import Image
import pytesseract

class TextExtractionModel:
    def __init__(self):
        # Explicitly set the path to the Tesseract executable
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    
    def extract_text(self, image_path):
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text
