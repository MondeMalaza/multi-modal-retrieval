import matplotlib.pyplot as plt
from PIL import Image
from src.index import search_index

def retrieve_images(index, query_features, image_paths, top_k=5):
    """Retrieves and displays the most relevant images"""
    indices = search_index(index, query_features, top_k)

    fig, axes = plt.subplots(1, top_k, figsize=(15, 5))
    for i, idx in enumerate(indices[0]):
        img = Image.open(image_paths[idx])
        axes[i].imshow(img)
        axes[i].axis('off')
        axes[i].set_title(f"Match {i+1}")
    plt.show()
