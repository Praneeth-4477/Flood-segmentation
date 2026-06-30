import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class FloodDataset(Dataset):
    def __init__(self, metadata_df, image_dir, mask_dir, transforms=None):
        self.metadata_df = metadata_df
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, self.metadata_df.iloc[idx, 0])
        mask_name = os.path.join(self.mask_dir, self.metadata_df.iloc[idx, 1])

        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Read mask as grayscale
        mask = cv2.imread(mask_name, cv2.IMREAD_GRAYSCALE)
        
        # Ensure mask and image have the same height and width
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Binarize mask to 0 and 1
        mask = (mask > 127).astype(np.float32)

        if self.transforms:
            augmented = self.transforms(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Add channel dimension to mask (1, H, W) if it's not present after transforms
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=0)

        return image, mask
