import io
import sys
import os
import tempfile
import uuid
import torch
import cv2
import numpy as np
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.table import Table
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import streamlit as st
from PIL import Image
from models.segmentation_model import SegmentationModel
from models.identification_model import IdentificationModel
from models.text_extraction_model import TextExtractionModel
from utils.visualization import visualize_segmented_objects
from transformers import DetrImageProcessor, DetrForObjectDetection
from utils.data_mapping import map_data_to_objects

# Load COCO labels
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

# Initialize models
segmentation_model = SegmentationModel()
identification_model = IdentificationModel(labels=coco_labels)
text_extraction_model = TextExtractionModel()
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")

# Title of the app
st.title("Image Processing Pipeline")

# File uploader to upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"], key="file_uploader_key")

if uploaded_file is not None:
    try:
        # Define the path to save the uploaded image in the 'data/input_images' folder
        input_images_dir = "data/input_images"
        os.makedirs(input_images_dir, exist_ok=True)
        input_image_path = os.path.join(input_images_dir, uploaded_file.name)
        
        # Save the uploaded file to the 'data/input_images' folder
        with open(input_image_path, "wb") as f:
            f.write(uploaded_file.read())

        # Display the uploaded image
        image = Image.open(input_image_path)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Processing...")

        # Convert image to numpy array and scale it for better detection
        image_np = np.array(image)
        height, width = image_np.shape[:2]
        scale_factor = 800 / max(height, width)  # Scale to a max dimension of 800 pixels
        new_dim = (int(width * scale_factor), int(height * scale_factor))
        scaled_image_np = cv2.resize(image_np, new_dim)
        scaled_image = Image.fromarray(scaled_image_np)

        # Perform object detection using Hugging Face model
        inputs = processor(images=scaled_image, return_tensors="pt")
        outputs = model(**inputs)

        # Post-process outputs
        target_sizes = torch.tensor([scaled_image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]  # Lower threshold

        # Scale back to original dimensions
        results["boxes"] = [[box[0] / scale_factor, box[1] / scale_factor, box[2] / scale_factor, box[3] / scale_factor] for box in results["boxes"]]

        # Display detected objects
        st.write("Identifying objects...")
        identified_objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [i.item() for i in box]  # Ensure box is a list of floats
            label_name = coco_labels[label.item()]  # Map label index to class name
            identified_objects.append({
                "label": label_name,
                "score": score.item(),
                "box": box
            })
            st.write(f"Object - Identified as: {label_name} with confidence {score:.2f}")

        # Visualize detected objects
        visualized_image = visualize_segmented_objects(
            image_np,
            [obj["box"] for obj in identified_objects],
            [coco_labels.index(obj["label"]) for obj in identified_objects],  # Convert label names to indices
            coco_labels
        )
        
        # Convert PIL image to bytes for Streamlit
        image_bytes = io.BytesIO()
        visualized_image.save(image_bytes, format='PNG')
        image_bytes.seek(0)
        
        st.image(image_bytes, caption='Detected Objects', use_column_width=True)

        # Extract text from the uploaded image
        st.write("Extracting text from the uploaded image...")
        extracted_text = text_extraction_model.extract_text(image_np)
        st.write(f"Extracted Text: {extracted_text}")

        # Step 2: Object Extraction, Storage, and Description/Summarization
        st.write("Extracting and generating summaries for identified objects...")
        master_id = str(uuid.uuid4())
        objects_dir = f"data/segmented_objects/{master_id}"
        os.makedirs(objects_dir, exist_ok=True)

        # Display segmented objects in a grid view
        cols = st.columns(3)  # Create three columns for grid layout
        object_data = []
        for i, obj in enumerate(identified_objects):
            label_name = obj["label"]
            score = obj["score"]
            box = obj["box"]
            box = [int(i) for i in box]
            obj_img = image_np[box[1]:box[3], box[0]:box[2]]
            
            if obj_img.size == 0:
                continue

            obj_pil = Image.fromarray(obj_img)
            obj_path = os.path.join(objects_dir, f"{label_name}_{i}.png")
            obj_pil.save(obj_path)
            
            # Display in grid layout
            with cols[i % 3]:
                st.image(obj_pil, caption=f"{label_name} - Score: {score:.2f}")
            
            # Generate and display description
            description = identification_model.generate_description(obj_img)
            st.write(f"Object Description: {description}")

            # Summarize each object
            summary = {
                "Object": label_name,
                "Confidence": f"{score:.2f}",
                "Description": description,
                "Bounding Box": box
            }
            st.write(f"Summarized Attributes of Each Object:")
            st.write(summary)

            object_data.append({
                "id": str(uuid.uuid4()),
                "label": label_name,
                "score": score,
                "box": box,
                "description": description
            })

        # Save data to file
        output_file = f"data/output/{master_id}_data_mapping.json"
        map_data_to_objects({"master_id": master_id, "extracted_text": extracted_text, "objects": object_data}, output_file)
        st.write(f"Data mapping saved to {output_file}")

        # Generate a summary table and final output image
        st.write("Generating final output image and summary table...")

        # Create DataFrame for the table
        df = pd.DataFrame(object_data)

        # Wrap text for the description column to prevent overflow
        df["description"] = df["description"].apply(lambda x: "\n".join([x[i:i+30] for i in range(0, len(x), 30)]))


        # Create a figure and axis for the table
        fig, ax = plt.subplots(figsize=(14, len(df) * 0.8 + 1))  # Adjust size based on number of rows
        ax.axis('off')  # Hide the axis

        # Create the table using plt.table for better control
        table = ax.table(
            cellText=df.values,
            colLabels=df.columns,
            cellLoc='center',
            loc='center',
            colColours=['lightgrey'] * len(df.columns),
            colWidths=[0.4, 0.2, 0.2, 0.2, 0.6] 
        )

        # Adjust table properties
        table.auto_set_font_size(False)
        table.set_fontsize(14)
        table.scale(1.2, 15)  # Scale the table: width=1, height=1.5 for better readability

        # Save the table as an image
        table_image_path = f"data/output/{master_id}_summary_table.png"
        plt.savefig(table_image_path, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)

        # Display the summary table
        table_image = Image.open(table_image_path)
        st.image(table_image, caption='Summary Table', use_column_width=True)
        st.write(f"Summary table saved to {table_image_path}")

        # Save final visualized image with annotations
        final_output_image_path = f"data/output/{master_id}_final_output.png"
        visualized_image.save(final_output_image_path)
        st.image(final_output_image_path, caption='Final Output Image with Annotations', use_column_width=True)
        st.write(f"Final output image saved to {final_output_image_path}")

    except Exception as e:
        st.error(f"An error occurred: {e}")