import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
# pyrefly: ignore [missing-import]
import albumentations as A
# pyrefly: ignore [missing-import]
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from dataset import FloodDataset
from model import create_model, DiceLoss, dice_coefficient

# Configuration
IMAGE_DIR = "Image"
MASK_DIR = "Mask"
METADATA_FILE = "metadata.csv"
BATCH_SIZE = 8
EPOCHS = 50
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BEST_MODEL_PATH = "best_model.pth"

def get_train_transforms():
    return A.Compose(
        [
            A.Resize(512, 512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

def get_val_transforms():
    return A.Compose(
        [
            A.Resize(512, 512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ]
    )

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader, desc="Training")
    model.train()
    
    total_loss = 0
    total_dice = 0
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        # Forward
        with torch.amp.autocast(device_type='cuda', enabled=True if DEVICE == "cuda" else False):
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update metrics
        total_loss += loss.item()
        with torch.no_grad():
            dice_score = dice_coefficient(predictions, targets).item()
            total_dice += dice_score
        
        # Update progress bar
        loop.set_postfix(loss=loss.item(), dice=dice_score)

    return total_loss / len(loader), total_dice / len(loader)

def check_accuracy(loader, model, loss_fn):
    total_loss = 0
    total_dice = 0
    model.eval()

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(DEVICE)
            targets = targets.to(DEVICE)
            
            predictions = model(data)
            loss = loss_fn(predictions, targets)
            
            total_loss += loss.item()
            dice_score = dice_coefficient(predictions, targets).item()
            total_dice += dice_score

    avg_loss = total_loss / len(loader)
    avg_dice = total_dice / len(loader)
    print(f"Validation Loss: {avg_loss:.4f} | Validation Dice: {avg_dice:.4f}")
    
    return avg_dice

def main():
    print(f"Using device: {DEVICE}")
    
    df = pd.read_csv(METADATA_FILE)
    
    # Split data 80% train / 20% validation
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    
    train_ds = FloodDataset(
        metadata_df=train_df,
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        transforms=get_train_transforms(),
    )

    val_ds = FloodDataset(
        metadata_df=val_df,
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        transforms=get_val_transforms(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=0, # keep 0 for Windows to avoid multiprocessing issues initially
        pin_memory=True,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=True,
        shuffle=False,
    )

    model = create_model().to(DEVICE)
    loss_fn = DiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    scaler = torch.amp.GradScaler()
    
    best_dice = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
        train_loss, train_dice = train_fn(train_loader, model, optimizer, loss_fn, scaler)
        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f}")
        
        val_dice = check_accuracy(val_loader, model, loss_fn)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            print(f"=> Saving new best model (Validation Dice: {best_dice:.4f})")
            torch.save(model.state_dict(), BEST_MODEL_PATH)

if __name__ == "__main__":
    main()
