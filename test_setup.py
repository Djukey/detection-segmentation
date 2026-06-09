# Test that all libraries import correctly
import torch
import torchvision
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO

print("All libraries imported successfully!")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# ============ STEP 2: LOAD AND PREPROCESS PASCAL VOC ============

# Check device (GPU vs CPU)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Image preprocessing: resize to 256x256 and normalize
image_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # Converts to PyTorch tensor & scales pixels to 0-1
])

# Mask preprocessing: resize only (don't normalize masks!)

mask_transform = transforms.Compose([
    transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.PILToTensor(),  # Keep mask values as class IDs
])

# Download and load Pascal VOC 2012 segmentation dataset

print("Downloading Pascal VOC 2012... (this will take 10-20 min the first time)")
dataset = datasets.VOCSegmentation(
    root="./data",
    year="2012",
    image_set="train",
    download=True,
    transform=image_transform,
    target_transform=mask_transform
)

# Create a DataLoader (feeds batches of images during training)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Dataset loaded! Number of training images: {len(dataset)}")
print(f"Batch size: 4")
