
from torchvision import transforms
import os
import random
from torchvision.transforms import functional as TF
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


# Define the transformation pipeline for input images (e.g., RGB photos)
input_transform = transforms.Compose([
    transforms.ToTensor(),  # Converts the PIL image to a PyTorch tensor and scales pixel values to [0, 1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalization values typically used for ImageNet-trained models
                         std=[0.229, 0.224, 0.225])   # Helps the model generalize better by standardizing inputs
])


# Define the transformation pipeline for segmentation masks (e.g., grayscale label maps)
mask_transform = transforms.Compose([
    transforms.PILToTensor(),  # Converts PIL image to tensor without scaling pixel values
    transforms.Lambda(lambda x: x.squeeze().long())  # Removes channel dimension (if present) and converts to long type (for class labels)
])


class SegmentationTransforms:
    """
    Data augmentation and preprocessing pipeline for image segmentation tasks.

    This transform class applies random transformations
    to both the input image and its corresponding segmentation mask,
    ensuring spatial alignment is preserved. The augmentations include
    random horizontal and vertical flips, affine transformations
    (rotation, scaling, shear), cropping, color jittering, optional Gaussian blur,
    and resizing.

    Parameters:
    -----------
    image_size : tuple of int (height, width), default=(256, 256)
        The output size for cropping and resizing the image and mask.
    
    degrees : float, default=15
        Maximum degrees for random rotation in the affine transformation.

    scale : tuple of float (min_scale, max_scale), default=(0.8, 1.2)
        Range of scale factors for random scaling in the affine transformation.

    shear : float, default=10
        Maximum shear angle (in degrees) for random shear in the affine transformation.

    Usage:
    ------
    transform = SegmentationTransforms(image_size=(256, 256), degrees=15, scale=(0.8, 1.2), shear=10)
    transformed_image, transformed_mask = transform(image, mask)

    Returns:
    --------
    image : PIL.Image or Tensor
        The augmented input image.

    mask : PIL.Image or Tensor
        The augmented segmentation mask, with nearest neighbor interpolation
        used for affine and resize operations to preserve label integrity.
    """
    def __init__(self, image_size=(256, 256), degrees=15, scale=(0.8, 1.2), shear=10):
        self.image_size = image_size
        self.degrees = degrees
        self.scale = scale
        self.shear = shear

    def __call__(self, image, mask):
        # Horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        # Vertical flip
        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        # Random affine transform (rotation + translation + scale + shear)
        angle = random.uniform(-self.degrees, self.degrees)
        translations = (0, 0)  # you can add translation if you want
        scale = random.uniform(self.scale[0], self.scale[1])
        shear = random.uniform(-self.shear, self.shear)
        image = TF.affine(image, angle=angle, translate=translations, scale=scale, shear=shear, fill=0)
        mask = TF.affine(mask, angle=angle, translate=translations, scale=scale, shear=shear, fill=0, interpolation=TF.InterpolationMode.NEAREST)

        # Random crop params
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.image_size)
        image = TF.crop(image, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)

        # Color jitter only on image
        image = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)(image)

        # Optional: random Gaussian blur on image only
        if random.random() > 0.7:
            image = TF.gaussian_blur(image, kernel_size=3)

        # Resize both to image_size (just in case)
        image = TF.resize(image, self.image_size)
        mask = TF.resize(mask, self.image_size, interpolation=TF.InterpolationMode.NEAREST)

        return image, mask


class SegmentationDataset(Dataset):
    """
    Custom dataset for image segmentation tasks.

    This dataset loads images and their corresponding segmentation masks
    from separate directories. It assumes that image and mask filenames
    match exactly to pair them correctly i.e. img_1

    Parameters:
    -----------
    image_dir : str
        Path to the directory containing input images.

    mask_dir : str
        Path to the directory containing segmentation masks.

    transform : callable, optional (default=None)
        A function/transform to apply to the input images.
        Typically used for data augmentation or preprocessing.

    target_transform : callable, optional (default=None)
        A function/transform to apply to the segmentation masks.
        Useful for mask-specific augmentations or preprocessing.

    Usage:
    ------
    dataset = SegmentationDataset(
        image_dir="path/to/images",
        mask_dir="path/to/masks",
        transform=image_transform,
        target_transform=mask_transform
    )

    image, mask = dataset[0]
    """
    def __init__(self, image_dir, mask_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = os.listdir(image_dir)
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        image = Image.open(os.path.join(self.image_dir, img_name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, img_name)).convert("L")

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)

        return image, mask


def delete_all_images(folder_path):
    """
    Deletes all image files in the specified folder.

    Supported image file extensions are: .png, .jpg, .jpeg, .bmp, .gif

    Parameters:
    -----------
    folder_path : str
        The path to the folder from which all image files will be deleted.

    Behavior:
    ---------
    - Iterates over all files in the folder.
    - Deletes files that match supported image extensions (case-insensitive).
    - Prints a confirmation message after deletion.

    Example:
    --------
    delete_all_images("/path/to/folder")
    """
    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.gif')
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(image_extensions):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    print(f"Deleted all images in {folder_path}")


def create_augmented_images(input_img_dir, input_mask_dir, output_img_dir, output_mask_dir, number_of_augmentor_images_per_example, transformer):
    """
    Generates augmented images and masks from the original dataset and saves them to output directories.

    This function:
    - Loads images and their corresponding masks from input directories.
    - Applies the given augmentation transform multiple times per original image.
    - Saves the augmented images and masks to specified output directories.
    - Clears the output directories before saving new augmented files.

    Parameters:
    -----------
    input_img_dir : str
        Directory path containing the original input images.

    input_mask_dir : str
        Directory path containing the corresponding segmentation masks.
        Assumes masks have the same base filename as images but with a ".png" extension.

    output_img_dir : str
        Directory path where augmented images will be saved.
        This folder will be cleared before saving.

    output_mask_dir : str
        Directory path where augmented masks will be saved.
        This folder will be cleared before saving.

    number_of_augmentor_images_per_example : int
        Number of augmented versions to generate per original image.

    transformer : callable
        A function or transform that takes an (image, mask) pair and returns
        an augmented (image, mask) pair. Expected to maintain spatial alignment.

    Behavior:
    ---------
    - Deletes all existing images in output directories before augmentation.
    - Supports input images with extensions .jpg, .jpeg, .png.
    - Saves augmented images and masks with incremental filenames: img_1.png, img_2.png, etc.
    - Uses tqdm to show progress bar during augmentation.

    Example:
    --------
    create_augmented_images(
        input_img_dir="data/images",
        input_mask_dir="data/masks",
        output_img_dir="data/aug_images",
        output_mask_dir="data/aug_masks",
        number_of_augmentor_images_per_example=5,
        transformer=my_segmentation_transform
    )
    """
    # List of original image files (excluding already augmented ones)
    image_filenames = [f for f in os.listdir(input_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    #Remove all current audmentations
    delete_all_images(output_img_dir)
    delete_all_images(output_mask_dir)

    count = 1
    for img_file in tqdm(image_filenames, desc="Augmenting"):
        base_name = os.path.splitext(img_file)[0]
        img_path = os.path.join(input_img_dir, img_file)
        mask_path = os.path.join(input_mask_dir, base_name + ".png")  # Assuming mask is .png

        # Load image and mask
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        for _ in range(number_of_augmentor_images_per_example):
            aug_img, aug_mask = transformer(image.copy(), mask.copy())

            aug_img.save(os.path.join(output_img_dir, f"img_{count}.png"))
            aug_mask.save(os.path.join(output_mask_dir, f"img_{count}.png"))

            count += 1


def show_example_training_data(output_img_dir, output_mask_dir):
    """
    Displays example image and mask pairs from the specified directories.

    This function loads the first 5 image-mask pairs from the given output
    directories and visualizes them side-by-side using matplotlib.

    Parameters:
    -----------
    output_img_dir : str
        Directory path containing images to display.

    output_mask_dir : str
        Directory path containing corresponding masks to display.
        It assumes filenames in this folder match those in output_img_dir.

    Behavior:
    ---------
    - Loads the first 5 images and their masks.
    - Displays images on the top row and corresponding masks on the bottom row.
    - Masks are displayed using a grayscale colormap.
    - Turns off axis ticks for clarity.

    Example:
    --------
    show_example_training_data("data/aug_images", "data/aug_masks")
    """
    # List of original image files (excluding already augmented ones)
    image_filenames = [f for f in os.listdir(output_img_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

    # Load the first 5 image-mask pairs
    images = [Image.open(os.path.join(output_img_dir, fname)) for fname in image_filenames[:5]]
    masks = [Image.open(os.path.join(output_mask_dir, fname)) for fname in image_filenames[:5]]

    # Plot
    fig, axs = plt.subplots(2, 5, figsize=(15, 6))

    for i in range(5):
        axs[0, i].imshow(images[i])
        axs[0, i].axis('off')
        axs[0, i].set_title(f"Image {i+1}")

        axs[1, i].imshow(masks[i], cmap='gray')
        axs[1, i].axis('off')
        axs[1, i].set_title(f"Mask {i+1}")

    plt.tight_layout()
    plt.show()


def train_model(epochs, model, train_loader, optimizer, criterion, device):
    """
    Trains a segmentation model for a specified number of epochs.

    Parameters:
    -----------
    epochs : int
        Number of times to iterate over the entire training dataset.

    model : torch.nn.Module
        The segmentation model to train.

    train_loader : torch.utils.data.DataLoader
        DataLoader providing batches of (image, mask) pairs for training.

    optimizer : torch.optim.Optimizer
        Optimizer used to update model weights.

    criterion : torch.nn.Module
        Loss function to compute the training loss (e.g., CrossEntropyLoss).

    device : torch.device
        Device on which to perform training (e.g., 'cpu' or 'cuda').

    Behavior:
    ---------
    - Sets the model to training mode.
    - Iterates over the data loader for the specified number of epochs.
    - For each batch:
        - Moves images and masks to the specified device.
        - Performs a forward pass to compute outputs.
        - Computes loss between outputs and masks.
        - Performs backpropagation and optimizer step.
    - Prints loss at the end of each epoch.

    Returns:
    --------
    model : torch.nn.Module
        The trained model after all epochs are completed.

    Example:
    --------
    trained_model = train_model(
        epochs=10,
        model=my_model,
        train_loader=my_train_loader,
        optimizer=my_optimizer,
        criterion=my_loss_fn,
        device=torch.device('cuda')
    )
    """
    model.train()
    for epoch in range(epochs):
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            output = model(imgs)['out']
            loss = criterion(output, masks)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}: Loss = {loss.item():.4f}")
    
    return model
