# Radon Angle Finder

This project is focused on developing a machine learning pipeline to automatically detect and measure the angles of image artifacts, primarily lines, within microscopic images of metal microstructures. The context for this work is the need to accurately determine the angle of strain within individual grains in metal microstructures, which is essential for understanding material behavior. Typically, the elements that reveal the strain are often lost or obscured by noise in the images, making manual detection difficult. Machine learning techniques can help mitigate this issue by enhancing feature detection, thereby improving both the speed and accuracy of strain analysis while reducing reliance on manual measurement.

## Overview

This approach leverages a pretrained machine learning model as a starting point, which is then fine-tuned on a targeted dataset of metal microstructure images to specialize the model for strain image detection. After training, the model segments the images to isolate relevant features corresponding to strain artifacts. Then, a Radon transform is applied to the segmented outputs to accurately extract the orientation angles of these linear features, enabling precise measurement of strain directions within the grains.

## The Project
The first image illustrates how a Radon transform generates a sinogram. It shows a series of four images depicting the step-by-step process: starting with the original image with a red arrow indicating the current angle of projection. The middle image displays the projection at that angle, and the right image shows the sinogram being built up step by step. The subsequent images demonstrate how these projections are stacked together to build the sinogram, which visually represents the image intensity of the image as a function of projection angle. It can be seen that when the angle of the projection is parrallel to the lines in an image, there is a strong intensity in the sinogram, and conversley, when the projection angle is perpendicular to the lines, the intensity is weakest. It is trivial at this point to detec the angles that have the highes tintensity, therefore the angle the imag artefacts lie at.
<p align="center">
  <img src="https://github.com/WilliamMAPearson/RadonAngleFinder/blob/main/src/readme_images/1%20radon_projections_output.png" width="500">
</p>





---

## Setup Instructions

This guide explains how to set up and run this project using a virtual environment and the provided `requirements.txt` file.

---

## 1. Check Python Version

This project works with python 3.13

## 2. Create Virtual Environment

Run the following in your terminal **from the root of the repo**:

```sh
python -m venv .venv
```

## 3. Activate Virtual Environment 

Run the following in your terminal **from the root of the repo**:

```sh
.venv\Scripts\activate
```

## 4. Ensure pip is up to date

Run the following in the terminal:

```sh
python.exe -m pip install --upgrade pip
```

## 5. Install Dependancies

Run the following in the terminal:

```sh
pip install -r requirements.txt
```

## 6. Run Test Files

Run the following in the terminal:

```sh
python main.py
```

## 7. Deactivate the Virtual Environment

Run the following in the terminal:

```sh
deactivate
```

## Exmaple

```sh
# Clone the repo
git clone https://github.com/your-username/your-repo.git
cd your-repo

# Create and activate venv
python -m venv .venv
source .venv/bin/activate

# Install packages
pip install -r requirements.txt

# Run the app
python main.py
```