# tests/test_identification.py
import unittest
from models.identification_model import IdentificationModel
from PIL import Image
import numpy as np

# Define COCO labels used for the identification model
coco_labels = [
    "N/A", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
    "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "TV", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

class TestIdentificationModel(unittest.TestCase):
    def setUp(self):
        self.model = IdentificationModel(labels=coco_labels)

    def test_generate_description(self):
        # Create a dummy image for testing
        image_np = np.zeros((224, 224, 3), dtype=np.uint8)
        image = Image.fromarray(image_np)

        description = self.model.generate_description(image)
        self.assertIsInstance(description, str)
        self.assertTrue(len(description) > 0)  # Ensure description is not empty

if __name__ == '__main__':
    unittest.main()
