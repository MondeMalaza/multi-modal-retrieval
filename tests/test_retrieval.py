import sys
import os
import torch
from unittest.mock import patch, MagicMock
from PIL import Image
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
from retrieval import retrieve_images  

@pytest.fixture
def mock_image_paths():
    """Returns a list of mock image paths."""
    return [f"mock_image_{i}.jpg" for i in range(10)]

@patch("src.index.search_index")
@patch("PIL.Image.open")
@patch("matplotlib.pyplot.show")
def test_retrieve_images(mock_plt_show, mock_image_open, mock_search_index, mock_image_paths):
    """Test retrieve_images function without displaying actual images."""

    # Mock search_index to return top 5 indices
    mock_search_index.return_value = [[0, 1, 2, 3, 4]]

    # Mock Image.open to return a blank image
    mock_image_open.return_value = Image.new("RGB", (224, 224))

    # Create a fake FAISS index and query features
    mock_index = MagicMock()
    query_features = torch.randn(1, 512)  # Fake feature vector

    # Call function
    retrieve_images(mock_index, query_features, mock_image_paths, top_k=5)

    # Assertions
    mock_search_index.assert_called_once_with(mock_index, query_features, 5)
    assert mock_image_open.call_count == 5  # Should open 5 images
    mock_plt_show.assert_called_once()  # Ensure plt.show() is called

    print("✅ test_retrieve_images passed!")

