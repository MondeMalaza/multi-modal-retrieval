import faiss
import numpy as np

def build_faiss_index(image_features):
    """Creates a FAISS index for fast image retrieval"""
    index = faiss.IndexFlatL2(image_features.shape[1])
    index.add(image_features.cpu().numpy())
    return index

def search_index(index, query_features, top_k=5):
    """Searches FAISS index for top-k matches"""
    distances, indices = index.search(query_features.cpu().numpy(), top_k)
    return indices
