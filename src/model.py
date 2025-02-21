import torch
import clip

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    """Loads the CLIP model"""
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def encode_images(model, images):
    """Encodes images using CLIP model"""
    with torch.no_grad():
        image_features = model.encode_image(images.to(device))
    return image_features / image_features.norm(dim=-1, keepdim=True)

def encode_text(model, text):
    """Encodes text query using CLIP model"""
    with torch.no_grad():
        text_features = model.encode_text(clip.tokenize([text]).to(device))
    return text_features / text_features.norm(dim=-1, keepdim=True)
