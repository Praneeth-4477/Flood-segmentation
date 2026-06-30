import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
# pyrefly: ignore [missing-import]
import albumentations as A
# pyrefly: ignore [missing-import]
from albumentations.pytorch import ToTensorV2

from model import create_model

# Configuration
BEST_MODEL_PATH = "best_model.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_inference_transforms():
    return A.Compose([
        A.Resize(512, 512),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def predict_image(image_path, model, transforms, device):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Keep original shape for later
    original_shape = image.shape[:2]
    
    # Apply transforms
    augmented = transforms(image=image)
    tensor_img = augmented['image'].unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        prediction = model(tensor_img)
        prediction = torch.sigmoid(prediction)
        prediction = (prediction > 0.5).float()
    
    # Squeeze and convert to numpy
    pred_mask = prediction.squeeze().cpu().numpy()
    
    # Resize mask back to original shape for overlay
    pred_mask = cv2.resize(pred_mask, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    return image, pred_mask

def visualize_prediction(image, pred_mask):
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(image)
    ax[0].set_title("Original Image")
    ax[0].axis("off")
    
    ax[1].imshow(pred_mask, cmap="gray")
    ax[1].set_title("Predicted Mask")
    ax[1].axis("off")
    
    # Overlay: create a red mask
    overlay = image.copy()
    overlay[pred_mask == 1] = [255, 0, 0]  # Color the flooded areas red
    
    # Blend with original image
    blended = cv2.addWeighted(image, 0.6, overlay, 0.4, 0)
    
    ax[2].imshow(blended)
    ax[2].set_title("Overlay")
    ax[2].axis("off")
    
    plt.tight_layout()
    plt.show()

def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python inference.py <path_to_image>")
        return

    image_path = sys.argv[1]
    
    print("Loading model...")
    model = create_model()
    
    try:
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE, weights_only=True))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model from {BEST_MODEL_PATH}. Make sure you run train.py first! ({e})")
        return
        
    model.to(DEVICE)
    transforms = get_inference_transforms()
    
    print(f"Predicting on {image_path}...")
    image, pred_mask = predict_image(image_path, model, transforms, DEVICE)
    
    print("Visualizing...")
    visualize_prediction(image, pred_mask)

if __name__ == "__main__":
    main()
