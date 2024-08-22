#utils/preprocessing.py
from PIL import Image

def resize_image(image_path, output_size=(800, 800)):
    image = Image.open(image_path)
    resized_image = image.resize(output_size)
    return resized_image
