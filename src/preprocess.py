import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi
from PIL import Image
import torch
import clip

api = KaggleApi()
api.authenticate()

KAGGLE_DATASET = "alessandrasala79/ai-vs-human-generated-dataset"
DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "kaggle_dataset")
IMAGE_FOLDER = os.path.join(DATASET_DIR, "test_data_v2")

os.makedirs(DATASET_DIR, exist_ok=True)

dataset_zip = os.path.join(DATASET_DIR, "dataset.zip")

if not os.path.exists(IMAGE_FOLDER):
    print("Downloading dataset from Kaggle...")
    api.dataset_download_files(KAGGLE_DATASET, path=DATASET_DIR, unzip=True)
    print("Dataset downloaded and extracted!")

def load_images(preprocess):
    """Loads and preprocesses images from IMAGE_FOLDER"""
    image_paths = [
        os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
        if f.endswith(('png', 'jpg', 'jpeg'))
    ][:500]

    image_tensors = [preprocess(Image.open(img)).unsqueeze(0) for img in image_paths]
    return image_paths, torch.cat(image_tensors)
