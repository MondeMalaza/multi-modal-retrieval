from flask import Flask, render_template, request
import torch
from src.preprocess import load_images
from src.model import load_model, encode_images, encode_text
from src.index import build_faiss_index
from src.retrieval import retrieve_images
import os

template_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../templates"))

app = Flask(__name__, template_folder=template_dir)

# Load CLIP model
model, preprocess = load_model()

# Load and preprocess images
image_paths, image_tensors = load_images(preprocess)

# Compute image embeddings
image_features = encode_images(model, image_tensors)

# Create FAISS index
index = build_faiss_index(image_features)

@app.route("/", methods=["GET", "POST"])
def home():
    retrieved_images = []
    
    if request.method == "POST":
        query = request.form["query"]
        query_features = encode_text(model, query)
        indices = retrieve_images(index, query_features, image_paths, top_k=5)

        # Get matching image paths
        retrieved_images = [image_paths[i] for i in indices[0]]

    return render_template("index.html", images=retrieved_images)

if __name__ == "__main__":
    app.run(debug=True)
