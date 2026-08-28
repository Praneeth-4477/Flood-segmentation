# Flood Area Image Segmentation

> **Deep learning–based semantic segmentation for detecting and precisely mapping flooded regions in satellite and ground-level imagery.**

A computer vision project built with **PyTorch** that uses a **U-Net architecture with a ResNet-34 encoder** to identify flooded regions at the pixel level.

Instead of simply classifying an image as *“flooded”* or *“not flooded,”* the model learns to determine **which exact pixels belong to the flooded area**, enabling precise flood-region mapping and visual analysis.

---

## Overview

Flood assessment from images is challenging because water can appear visually similar to roads, buildings, shadows, and other land surfaces.

This project approaches the problem as a **binary semantic segmentation task**:

```text
Input Image
     │
     ▼
ResNet-34 Encoder
     │
     ▼
Feature Extraction
     │
     ▼
U-Net Decoder
     │
     ▼
Pixel-wise Prediction
     │
     ▼
Flood Segmentation Mask
     │
     ▼
Visual Overlay
```

The resulting segmentation mask highlights the predicted flooded regions, making the model's output directly interpretable.

---

# Architecture & Approach

### U-Net + ResNet-34

The model combines the proven **U-Net encoder-decoder architecture** with a pretrained **ResNet-34 backbone**.

* **Encoder — ResNet-34**

  * Extracts hierarchical visual features from the input image.
  * Captures both low-level spatial information and high-level semantic features.

* **Decoder — U-Net**

  * Reconstructs the spatial resolution of the extracted features.
  * Uses skip connections to preserve fine-grained spatial information.
  * Produces a pixel-level flood probability map.

This combination provides a strong balance between **feature extraction capability and precise spatial segmentation**.

---

## Loss Function

### Custom Dice Loss

Flooded regions can occupy a relatively small portion of an image, making conventional pixel-wise losses less effective in some cases.

To address this, the project uses a custom **Dice Loss**, optimized around the Dice coefficient.

The Dice coefficient measures the overlap between the predicted flood mask and the ground-truth mask:

```text
                 2 × |Prediction ∩ Ground Truth|
Dice Score = ─────────────────────────────────────
                 |Prediction| + |Ground Truth|
```

Higher Dice scores indicate stronger overlap between predicted and actual flooded regions.

---

# Data Augmentation

To improve generalization across different environments and image conditions, the training pipeline uses **Albumentations**.

The augmentation pipeline includes:

* Horizontal flips
* Vertical flips
* Brightness adjustments
* Contrast adjustments
* Image resizing
* Normalization

All images are resized to:

**512 × 512**

This helps the model become more robust to variations in **orientation, lighting, contrast, and visual conditions**.

---

# Project Structure

```text
Flood-segmentation/
│
├── dataset.py
│   └── Dataset class
│       ├── Image loading
│       ├── Mask loading
│       └── Data augmentation
│
├── model.py
│   ├── U-Net + ResNet-34
│   ├── Dice Loss
│   └── Dice Coefficient
│
├── train.py
│   ├── Training loop
│   ├── Validation loop
│   ├── Metric tracking
│   └── Best model checkpointing
│
├── inference.py
│   ├── Model loading
│   ├── Image preprocessing
│   ├── Flood prediction
│   └── Segmentation overlay
│
├── requirements.txt
│
└── README.md
```

---

# Training Pipeline

The training process follows a standard supervised segmentation workflow:

```text
Raw Images + Ground Truth Masks
              │
              ▼
       Data Augmentation
              │
              ▼
          512 × 512
              │
              ▼
       ResNet-34 Encoder
              │
              ▼
        U-Net Decoder
              │
              ▼
      Flood Probability Map
              │
              ▼
          Dice Loss
              │
              ▼
       Backpropagation
              │
              ▼
          Validation
              │
              ▼
     Best Model Checkpoint
```

The model is trained for **50 epochs**, with the best-performing checkpoint saved as:

```text
best_model.pth
```

---

# Inference

Once trained, the model can be used to segment a new image.

```bash
python inference.py Image/0.jpg
```

The inference pipeline:

1. Loads the trained model.
2. Preprocesses the input image.
3. Generates a pixel-level flood prediction.
4. Produces a segmentation mask.
5. Creates an overlay showing the detected flood region.

This makes it possible to visually compare:

```text
Original Image
       +
Predicted Flood Mask
       ↓
Flooded Area Overlay
```

---

# Tech Stack

| Component         | Technology     |
| ----------------- | -------------- |
| Deep Learning     | PyTorch        |
| Architecture      | U-Net          |
| Encoder           | ResNet-34      |
| Loss Function     | Dice Loss      |
| Data Augmentation | Albumentations |
| Image Processing  | PyTorch / PIL  |
| Language          | Python         |
| Model Checkpoint  | `.pth`         |

---

# Key Features

* Pixel-level flood detection
* U-Net semantic segmentation
* ResNet-34 feature extraction
* Custom Dice Loss optimization
* Robust image augmentation
* 512×512 input pipeline
* Validation-based model checkpointing
* Standalone inference pipeline
* Visual flood-mask overlays
* GPU/CUDA compatible training

---

# Why Segmentation?

Traditional image classification answers:

> **“Is there a flood in this image?”**

This project goes further:

> **“Which exact pixels represent the flooded region?”**

That distinction makes semantic segmentation significantly more useful for downstream applications such as:

* Flood extent estimation
* Disaster response
* Damage assessment
* Satellite imagery analysis
* Emergency mapping
* Post-disaster monitoring
* Geographic risk analysis

---

# Future Improvements

Potential extensions include:

* Multi-class segmentation for water, buildings, roads, and vegetation
* Larger and more diverse satellite-image datasets
* Attention-based U-Net architectures
* Transformer-based segmentation models
* Real-time inference
* Automated flood-area estimation
* GIS integration
* Geo-referenced flood maps
* Deployment as a web-based inference API

---

# Getting Started

## 1. Clone the repository

```bash
git clone https://github.com/Praneeth-4477/Flood-segmentation.git
cd Flood-segmentation
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

> For GPU acceleration, install a CUDA-compatible version of PyTorch appropriate for your system.

## 3. Train the model

```bash
python train.py
```

The training process runs for **50 epochs** and saves the best validation checkpoint as:

```text
best_model.pth
```

## 4. Run inference

```bash
python inference.py Image/0.jpg
```

---

# Project Pipeline

```text
              ┌─────────────────┐
              │    Input Image  │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Preprocessing  │
              │   + Augmentation│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   ResNet-34      │
              │     Encoder      │
              └────────┬────────┘
                       │
                 Feature Maps
                       │
                       ▼
              ┌─────────────────┐
              │   U-Net Decoder │
              │ + Skip Connections│
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │ Flood Probability│
              │      Mask        │
              └────────┬────────┘
                       │
                       ▼
              ┌─────────────────┐
              │  Flood Overlay  │
              └─────────────────┘
```

---

## Project Goal

The goal of this project is to demonstrate how **deep learning and semantic segmentation can transform raw imagery into actionable flood-region information**.

Rather than treating an image as a single label, the model learns the **spatial structure of the disaster itself**—identifying where flooding occurs at the pixel level.

---

## License

This project is intended for educational and research purposes.
