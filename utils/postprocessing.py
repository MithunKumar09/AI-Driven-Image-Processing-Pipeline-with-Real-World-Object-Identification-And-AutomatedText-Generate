#utils/postprocessing.py
import os
from PIL import Image

def save_segmented_objects(prediction, image_path, output_dir):
    image = Image.open(image_path)
    os.makedirs(output_dir, exist_ok=True)
    for i, mask in enumerate(prediction[0]['masks']):
        obj_image = Image.fromarray((mask[0] * 255).byte().cpu().numpy())
        obj_image.save(os.path.join(output_dir, f"object_{i+1}.png"))
