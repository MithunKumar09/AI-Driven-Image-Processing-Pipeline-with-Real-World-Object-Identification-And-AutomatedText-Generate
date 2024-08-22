# utils/visualization.py
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch

def visualize_segmented_objects(image_np, boxes, labels, coco_labels):
    """
    Visualize the segmented objects on the image.
    
    :param image_np: Numpy array of the image
    :param boxes: List of bounding boxes for detected objects
    :param labels: List of labels for detected objects
    :param coco_labels: List of COCO class labels
    :return: PIL Image with visualized objects
    """
    # Convert numpy array image to PIL Image
    image_pil = Image.fromarray(image_np)
    draw = ImageDraw.Draw(image_pil)
    
    # Define font for text (adjust the path to a font file as needed)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    for box, label in zip(boxes, labels):
        # Ensure that box coordinates are in Python list or tuple
        if isinstance(box, torch.Tensor):
            box = box.tolist()
        
        # Convert box coordinates to integers and round them
        box = [int(round(coord)) for coord in box]
        
        # Draw bounding box
        draw.rectangle(box, outline="red", width=3)
        
        # Draw label text with background for better readability
        label_name = coco_labels[label]
        
        # Calculate text size
        text_bbox = draw.textbbox((box[0], box[1] - 20), label_name, font=font)
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        
        # Define the position of the text
        text_x = box[0]
        text_y = box[1] - text_size[1] - 5
        
        # Ensure text is within image boundaries
        if text_y < 0:
            text_y = box[1] + 5
        
        # Draw background rectangle for text
        draw.rectangle([text_x, text_y, text_x + text_size[0], text_y + text_size[1]], fill="red")
        
        # Draw text on top of the background rectangle
        draw.text((text_x, text_y), label_name, fill="white", font=font)
    
    return image_pil

