# models/identification_model.py
import numpy as np
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM

class IdentificationModel:
    def __init__(self, labels):
        self.coco_labels = labels
        # Initialize CLIP model for object identification
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Initialize text generation model (GPT-2 or GPT-Neo)
        self.text_generation_model = "EleutherAI/gpt-neo-1.3B"  # Can be switched to "gpt2"
        self.tokenizer = AutoTokenizer.from_pretrained(self.text_generation_model)
        self.text_generator = AutoModelForCausalLM.from_pretrained(self.text_generation_model)
                # Initialize GPT-Neo model for summarization
        self.summarization_model = "EleutherAI/gpt-neo-1.3B"
        self.tokenizer = AutoTokenizer.from_pretrained(self.summarization_model)
        self.summarizer = AutoModelForCausalLM.from_pretrained(self.summarization_model)


        # Ensure pad_token_id is correctly set
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def generate_description(self, image):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        # Transform the image for CLIP model
        inputs = self.clip_processor(images=image, return_tensors="pt")
        outputs = self.clip_model.get_image_features(**inputs)

        identified_objects = []
        for label in self.coco_labels[1:]:  # Skip the first 'N/A' label
            text_inputs = self.clip_processor(text=[f"This is a {label}"], return_tensors="pt")
            text_features = self.clip_model.get_text_features(**text_inputs)
            similarity = torch.cosine_similarity(outputs, text_features, dim=-1).item()

            if similarity > 0.3:  # Threshold for identification
                description = self.get_description(label)
                identified_objects.append(description)

        return ". ".join(identified_objects) if identified_objects else "No relevant objects could be described."

    def get_description(self, label):
        prompt = f"Define {label} in the real world."

        try:
            # Encode the prompt and generate text
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.text_generator.generate(
                **inputs,
                max_length=100,  # Reduced max_length for concise definitions
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                pad_token_id=self.tokenizer.eos_token_id,  # Ensure proper padding
                eos_token_id=self.tokenizer.eos_token_id  # Ensure proper end-of-sequence token
            )

            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # Ensure that the text ends with a proper punctuation
            if not generated_text or generated_text.lower() == prompt.lower():
                return f"Definition of {label} is unavailable."
            
            # Add a period at the end if not present
            if not generated_text.endswith('.'):
                generated_text += '.'

            return generated_text

        except Exception as e:
            print(f"Error generating definition for {label}: {e}")
            return f"Definition for {label} could not be generated due to an error."
        
    def generate_summary(self, text):
        prompt = f"Summarize the following process in approximately 150 words: {text}"

        try:
            # Encode the prompt and generate text
            inputs = self.tokenizer(prompt, return_tensors="pt")
            outputs = self.summarizer.generate(
                **inputs,
                max_length=200,  # Set max_length to cover detailed summaries
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95
            )

            summary_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

            # Ensure that the summary ends with a proper punctuation
            if not summary_text.endswith('.'):
                summary_text += '.'

            return summary_text

        except Exception as e:
            print(f"Error generating summary: {e}")
            return "Summary could not be generated due to an error."
        
    def is_quota_exceeded(self):
        # Since we're running locally, no quota checks are needed for GPT-Neo or GPT-2
        return False
