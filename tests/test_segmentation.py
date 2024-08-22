# tests/test_segmentation.py
import unittest
from models.segmentation_model import SegmentationModel
from PIL import Image
import numpy as np
import os

class TestSegmentationModel(unittest.TestCase):
    def setUp(self):
        self.model = SegmentationModel()
        self.image = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        self.output_dir = "data/segmented_objects"
        os.makedirs(self.output_dir, exist_ok=True)

    def test_segmentation(self):
        prediction = self.model.segment(self.image)
        self.assertIsInstance(prediction, dict)
        self.assertIn('masks', prediction)
    
    def test_save_segmented_objects(self):
        prediction = self.model.segment(self.image)
        num_objects = self.model.save_segmented_objects(prediction, None, self.output_dir)
        self.assertTrue(num_objects > 0)
        for i in range(num_objects):
            self.assertTrue(os.path.exists(f"{self.output_dir}/object_{i}.png"))

if __name__ == '__main__':
    unittest.main()
