import os
import torch
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
import clip
from unittest import mock
from unittest.mock import patch, MagicMock
from PIL import Image
from preprocess import load_images

# Mock the preprocess function from CLIP
@pytest.fixture
def mock_preprocess():
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    return preprocess

# Mock the image loading function
@patch("os.listdir")
@patch("PIL.Image.open")
def test_load_images(mock_image_open, mock_listdir, mock_preprocess):
    """Test that load_images correctly loads and processes images."""
    
    # Simulate 500+ image files in the directory
    mock_listdir.return_value = [f"image_{i}.jpg" for i in range(600)]

    # Mock PIL Image.open to return a fake image
    fake_image = Image.new("RGB", (224, 224))  # Create a dummy image
    mock_image_open.return_value = fake_image

    # Call the function
    image_paths, image_tensors = load_images(mock_preprocess)

    # Check if exactly 500 images are loaded
    assert len(image_paths) == 500, "Should load exactly 500 images"
    
    # Check if the tensor shape matches (500, 3, H, W)
    assert image_tensors.shape[0] == 500, "Tensor batch size should be 500"

    # Check if each tensor has 3 color channels
    assert image_tensors.shape[1] == 3, "Each image should have 3 color channels (RGB)"

    print("✅ test_load_images passed!")

