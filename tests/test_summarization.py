# tests/test_summarization.py
import unittest
from models.summarization_model import SummarizationModel

class TestSummarizationModel(unittest.TestCase):
    def setUp(self):
        self.model = SummarizationModel()
        self.text = "This is a detailed description of an object processing pipeline. The pipeline involves uploading an image, performing object detection, extracting text, and generating descriptions."

    def test_summarization(self):
        summary = self.model.summarize(self.text)
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 0)
        self.assertLessEqual(len(summary.split()), 150)  # Ensure the summary is approximately 150 words

if __name__ == '__main__':
    unittest.main()

