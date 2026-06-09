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





#  STEP 2: LOAD AND PREPROCESS PASCAL VOC 

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


# TEST HERE, BIG TEST TO VERIFY: Show one image + its mask 

# Grab the first image and mask
#image, mask = dataset[0]

# Convert tensors to numpy for matplotlib
#image_np = image.permute(1, 2, 0).numpy()  # Reorder: (channels, h, w) → (h, w, channels)
#mask_np = mask.squeeze().numpy()  # Remove single channel dim

# Display side by side
#fig, axes = plt.subplots(1, 2, figsize=(12, 5))
#axes[0].imshow(image_np)
#axes[0].set_title("Original Image")
#axes[0].axis("off")

#axes[1].imshow(mask_np, cmap="tab20")  # tab20 = colorful colormap for classes
#axes[1].set_title("Segmentation Mask")
#axes[1].axis("off")

#plt.tight_layout()
#plt.show()

# STEP 3: YOLO FOR OBJECT DETECTION 

from ultralytics import YOLO
from PIL import Image

# Load pretrained YOLOv8 model (nano version - small and fast)
print("\nLoading YOLOv8 nano model...")
yolo_model = YOLO("yolov8n.pt")  # Downloads automatically first time (~6 MB)

# Get the first image from dataset (as PIL image for YOLO)
image_tensor, mask_tensor = dataset[0]

# Convert tensor back to PIL image for YOLO
image_pil = transforms.ToPILImage()(image_tensor)

# Run YOLO inference on the image
print("Running YOLO detection...")
yolo_results = yolo_model(image_pil)

# YOLO returns results - let's visualize them
# .plot() returns the image with bounding boxes drawn on it
annotated_image = yolo_results[0].plot()

# Display
plt.figure(figsize=(12, 7))
plt.imshow(annotated_image[:, :, ::-1])  # Convert BGR → RGB for matplotlib
plt.title("YOLO Object Detection — Bounding Boxes")
plt.axis("off")
plt.show()

# Print what YOLO detected
print("\n--- YOLO Detection Results ---")
for box in yolo_results[0].boxes:
    class_id = int(box.cls[0])
    class_name = yolo_model.names[class_id]
    confidence = float(box.conf[0])
    print(f"  Detected: {class_name} (confidence: {confidence:.2%})")



    # STEP 4: U-NET ARCHITECTURE

import torch.nn as nn
import torch.optim as optim

# Helper: a "double convolution" block (used many times in U-Net)
class DoubleConv(nn.Module):
    """Two consecutive Conv+BatchNorm+ReLU layers — the basic building block."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net architecture for image segmentation.
    
    Shape: encoder (shrink) → bottleneck → decoder (expand)
    with skip connections preserving detail.
    """
    def __init__(self, in_channels=3, num_classes=21):
        # Pascal VOC has 21 classes: 20 object types + 1 background
        super().__init__()
        
        # ENCODER (downsampling path) - shrinks image, learns features
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        
        # BOTTLENECK (lowest point of the U)
        self.bottleneck = DoubleConv(512, 1024)
        
        # DECODER (upsampling path) - expands back to full size
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)  # 1024 because of skip connection
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)
        
        # FINAL OUTPUT layer - produces the per-pixel class predictions
        self.final = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # Downsampling operation used between encoder blocks
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # ENCODER path (save each output for skip connections)
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        
        # BOTTLENECK
        b = self.bottleneck(self.pool(e4))
        
        # DECODER path (with skip connections from encoder)
        d4 = self.upconv4(b)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))  # concatenate skip
        
        d3 = self.upconv3(d4)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.upconv2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.upconv1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # OUTPUT
        return self.final(d1)


# Build the model and move to GPU
print("\nBuilding U-Net model...")
unet_model = UNet(in_channels=3, num_classes=21).to(device)

# Count parameters (gives a sense of model size)
total_params = sum(p.numel() for p in unet_model.parameters())
print(f"U-Net created! Total parameters: {total_params:,}")
print(f"Model is on: {next(unet_model.parameters()).device}")






# STEP 5: TRAIN THE U-NET

from tqdm import tqdm

# Loss function - CrossEntropyLoss with ignore_index=255 
# (skip the "boundary/unknown" pixels that Pascal VOC marks with 255)
criterion = nn.CrossEntropyLoss(ignore_index=255)

# Optimizer - Adam is a reliable default for most deep learning
optimizer = optim.Adam(unet_model.parameters(), lr=1e-4)

# Number of training epochs
NUM_EPOCHS = 5

print(f"\nStarting training for {NUM_EPOCHS} epochs...")
print(f"Total batches per epoch: {len(dataloader)}")
print(f"This will take approximately {NUM_EPOCHS * 8} minutes on your GPU.\n")

# Training loop
unet_model.train()  # Set model to training mode

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    
    # tqdm wraps the dataloader to show a progress bar
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
    
    for images, masks in progress_bar:
        # Move data to GPU
        images = images.to(device)
        # Masks need to be long type and the channel dim removed
        masks = masks.squeeze(1).long().to(device)
        
        # Forward pass: get predictions from U-Net
        outputs = unet_model(images)
        
        # Calculate loss (how wrong were we?)
        loss = criterion(outputs, masks)
        
        # Backward pass: compute gradients
        optimizer.zero_grad()  # Reset gradients
        loss.backward()        # Compute new gradients
        optimizer.step()       # Update model weights
        
        # Track loss
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    
    avg_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")

# Save the trained model for later use
torch.save(unet_model.state_dict(), "unet_voc_trained.pth")
print("\nTraining complete! Model saved as 'unet_voc_trained.pth'")



# STEP 6: COMBINE YOLO + U-NET

print("\n\nRunning combined YOLO + U-Net pipeline...")

# Pascal VOC class names (in order, 0=background through 20=tvmonitor)
VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
    'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
    'train', 'tvmonitor'
]

# Set U-Net to evaluation mode (turns off training-only behaviors)
unet_model.eval()

# Grab a test image from the dataset
test_image_tensor, test_mask_tensor = dataset[0]
test_image_pil = transforms.ToPILImage()(test_image_tensor)

# --- Run YOLO to detect objects ---
yolo_results = yolo_model(test_image_pil)
detection_boxes = yolo_results[0].boxes

# --- Run U-Net on the full image to get segmentation ---
with torch.no_grad():  # No gradients needed for inference (saves memory)
    # Add batch dimension and move to GPU
    image_for_unet = test_image_tensor.unsqueeze(0).to(device)
    unet_output = unet_model(image_for_unet)
    # Get the class with highest score for each pixel
    predicted_mask = unet_output.argmax(dim=1).squeeze(0).cpu().numpy()

# --- Visualize everything together ---
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Panel 1: Original image
axes[0].imshow(test_image_tensor.permute(1, 2, 0).numpy())
axes[0].set_title("Original Image")
axes[0].axis("off")

# Panel 2: YOLO detection with bounding boxes
yolo_annotated = yolo_results[0].plot()
axes[1].imshow(yolo_annotated[:, :, ::-1])  # BGR → RGB
axes[1].set_title("YOLO Object Detection")
axes[1].axis("off")

# Panel 3: U-Net segmentation
axes[2].imshow(test_image_tensor.permute(1, 2, 0).numpy())
axes[2].imshow(predicted_mask, cmap="tab20", alpha=0.5)  # Overlay mask with transparency
axes[2].set_title("U-Net Segmentation Overlay")
axes[2].axis("off")

plt.tight_layout()
plt.savefig("final_result.png", dpi=150, bbox_inches="tight")
print("Final visualization saved as 'final_result.png'")
plt.show()

# Print what each model found
print("\n--- YOLO detected: ---")
for box in detection_boxes:
    class_id = int(box.cls[0])
    class_name = yolo_model.names[class_id]
    confidence = float(box.conf[0])
    print(f"  {class_name} (confidence: {confidence:.2%})")

print("\n--- U-Net segmented these VOC classes: ---")
unique_classes = np.unique(predicted_mask)
for class_id in unique_classes:
    if class_id < len(VOC_CLASSES):
        print(f"  {VOC_CLASSES[class_id]}")

print("\n🎉 Yah Yah Yah Prof Chris Pipeline complete! YOLO + U-Net working together.")

