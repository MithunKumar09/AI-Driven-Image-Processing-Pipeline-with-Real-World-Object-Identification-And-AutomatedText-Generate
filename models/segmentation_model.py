#models/segmentation_model.py
import numpy as np
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image

class SegmentationModel:
    def __init__(self):
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval()

    def segment(self, image):
        # Ensure the image is a PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        image_tensor = F.to_tensor(image).unsqueeze(0)
        with torch.no_grad():
            prediction = self.model(image_tensor)[0]
        return prediction

    def save_segmented_objects(self, prediction, image_path, output_dir):
        import cv2
        import os

        image = Image.open(image_path)
        image_np = np.array(image)

        objects_saved = 0
        for i, mask in enumerate(prediction['masks']):
            mask = mask[0].mul(255).byte().cpu().numpy()
            mask = np.stack([mask] * 3, axis=-1)  # Make it 3-channel
            masked_image = np.where(mask, image_np, 0)
            filename = os.path.join(output_dir, f"object_{i}.png")
            cv2.imwrite(filename, cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
            objects_saved += 1
        
        return objects_saved

