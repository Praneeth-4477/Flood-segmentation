# Flood Area Image Segmentation

This project implements a deep learning model to segment flooded areas in images using PyTorch.

## Architecture and Approach
- **Model:** U-Net architecture with a ResNet-34 encoder backbone.
- **Loss Function:** Custom Dice Loss, optimizing for the Dice Coefficient metric.
- **Data Augmentation:** Used Albumentations for robust data augmentation, including horizontal/vertical flips, brightness/contrast adjustments, and resizing images to 512x512.

## Files
- `dataset.py`: Defines the PyTorch Dataset class, handles image/mask loading, and applies transforms.
- `model.py`: Defines the U-Net model creation, DiceLoss function, and dice_coefficient metric.
- `train.py`: Contains the training loop, validation loop, and saves the best model based on the validation score.
- `inference.py`: Script to load the trained model, run prediction on a new image, and display an overlay.

## Setup and Usage

1. **Install dependencies:**
   `pip install -r requirements.txt`
   (Note: For GPU acceleration, ensure PyTorch with CUDA is installed).

2. **Training the model:**
   `python train.py`
   This trains the model for 50 epochs and saves `best_model.pth`.

3. **Running inference:**
   `python inference.py Image/0.jpg`
