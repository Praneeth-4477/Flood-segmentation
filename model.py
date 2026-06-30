import torch
import torch.nn as nn
# pyrefly: ignore [missing-import]
import segmentation_models_pytorch as smp

def create_model(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3, classes=1):
    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=classes,
    )
    return model

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)  
        
        return 1.0 - dice

def dice_coefficient(inputs, targets, smooth=1.0):
    inputs = torch.sigmoid(inputs)
    inputs = (inputs > 0.5).float() # Threshold to get binary mask
    
    # Flatten label and prediction tensors
    inputs = inputs.view(-1)
    targets = targets.view(-1)
    
    intersection = (inputs * targets).sum()                            
    dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)  
    
    return dice
