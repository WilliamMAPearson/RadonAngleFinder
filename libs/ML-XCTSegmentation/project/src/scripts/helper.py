import torch

from scripts.train import *
from scripts.test import *

from PIL import Image


def pre_process(image_path):
    """
    Loads an image from the specified path, applies preprocessing transformations,
    and returns a batch-ready tensor suitable for model input.

    Parameters:
    -----------
    image_path : str
        Path to the input image file.

    Returns:
    --------
    torch.Tensor
        Preprocessed image tensor with shape (1, 3, H, W), where 1 is the batch size,
        3 is the number of color channels (RGB), and H, W are the height and width.

    Notes:
    ------
    - Assumes 'input_transform' is defined elsewhere and applies the necessary
      normalization, resizing, and tensor conversion.
    - Converts the image to RGB before applying the transform.
    """
    image = Image.open(image_path).convert("RGB")  # Ensure the image is in RGB format
    return input_transform(image).unsqueeze(0)     # Add a batch dimension (required by most models)


def segment_with_model(image_path, model, device):
    #Load image
    image = Image.open(image_path).convert("RGB")

    #Force model into eval
    model.eval()

    # Read in an Example Image
    input_tensor = pre_process(image_path).to(device)

    #Attempt to analyse the example image
    with torch.no_grad():
        output = model(input_tensor)['out']
        predicted = output.argmax(1).squeeze(0).cpu().numpy()

    return image, predicted


def plot_segment_image_with_model(image_path, model, device):
    #Segment with model
    image, predicted = segment_with_model(image_path, model, device)

    #Visualise the sementation
    visualize(image, predicted)


def plot_segmented_images(folder_path, model, device, count=5):
    """
    Plots a specified number of validation images in a single figure, showing original and segmented results.

    Parameters:
    -----------
    folder_path : str
        Path to the folder containing PNG images.

    model : torch.nn.Module
        Trained segmentation model.

    device : torch.device
        Device on which to run the model.

    count : int, optional (default=5)
        Number of images to process and plot (min=1, max=5).

    Behavior:
    ---------
    - Loads up to `count` images from the folder.
    - Segments each image and creates a 2-row plot:
        Top row: Original images
        Bottom row: Segmentation predictions.
    """
    count = max(1, min(count, 5))  # Enforce bounds

    image_files = []

    # Collect up to `count` valid image paths
    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            if file.lower().endswith(".png"):
                image_files.append(os.path.join(root, file))
                if len(image_files) == count:
                    break

    #Rest count incase its smaller
    count = len(image_files)

    # Prepare figure with dynamic columns
    fig, axs = plt.subplots(2, count, figsize=(3 * count, 8))
    fig.suptitle("Validation vs Segmented Images", fontsize=16)

    # Handle axs shape when count == 1
    if count == 1:
        axs = axs.reshape(2, 1)

    for i, image_path in enumerate(image_files):
        image, predicted = segment_with_model(image_path, model, device)

        # Optional: define color palette
        palette = np.array([
            [0, 0, 0],         # class 0: background
            [255, 0, 0],       # class 1: your object (e.g., red)
            # Add more colors if more classes
        ])

        # Create RGB segmentation map
        color_mask = palette[predicted]

        axs[0, i].imshow(image)
        axs[0, i].set_title(f"Original {i+1}")
        axs[0, i].axis("off")

        axs[1, i].imshow(color_mask)
        axs[1, i].set_title(f"Segmented {i+1}")
        axs[1, i].axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()