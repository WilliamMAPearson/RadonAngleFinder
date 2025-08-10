import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def visualize(image, prediction):
    # Optional: define color palette
    palette = np.array([
        [0, 0, 0],         # class 0: background
        [255, 0, 0],       # class 1: your object (e.g., red)
        # Add more colors if more classes
    ])

    # Create RGB segmentation map
    color_mask = palette[prediction]

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Predicted Segmentation")
    plt.imshow(color_mask)
    plt.axis('off')

    plt.show()