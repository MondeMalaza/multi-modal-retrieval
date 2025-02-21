import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

import pytest
from web_app import app

@pytest.fixture
def client():
    """Set up Flask test client."""
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client

def test_homepage(client):
    """Test if homepage loads correctly."""
    response = client.get("/")
    assert response.status_code == 200

def test_image_search(client, mocker):
    """Test image search endpoint."""
    mock_retrieval = mocker.patch("src.web_app.retrieve_images", return_value=["image1.jpg", "image2.jpg"])
    
    response = client.post("/search", json={"query": "a cat sitting on a chair"})
    
    assert response.status_code == 200
    assert "image1.jpg" in response.json["images"]
