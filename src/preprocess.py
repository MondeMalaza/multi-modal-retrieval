import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import Image
import torch
import clip

# Ensure script runs from correct directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_FOLDER = os.path.join(SCRIPT_DIR, "..", "test_data_v2_sampled")

def load_images(preprocess):
    """Loads and preprocesses images from IMAGE_FOLDER"""
    image_paths = [
        os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER) 
        if f.endswith(('png', 'jpg', 'jpeg'))
    ][:500]  # Sample 500 images

    image_tensors = [preprocess(Image.open(img)).unsqueeze(0) for img in image_paths]
    return image_paths, torch.cat(image_tensors)
