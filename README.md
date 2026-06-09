# detection-segmentation

Object detection &amp; Segmentation: Combining YOLO + U-Net on Pascal VOC

**What this project is all about** 

This project is an AI-powered surveillance prototype that I enjoyed building for my week 4 in computer vision class at Concordia University. Basically, the tool combines deep learning models to enable the system to detect objects in a dataset and outline their exact pixel locations. 

YOLO (I picked YOLOv8) performs object detection by bounding boxes around objects it recognizes. U-Net, naturally, handles segmentation.

**What happens when you run the script**

Loads the Pascal VOC 2012 dataset (1464 training images, 21 classes).
Runs a quick YOLO detection on a sample image
Builds a U-Net on the dataset for 5 epochs
Combines U-Net and YOLO together, showing a 3-panel result.
Then, at the end, it saves the final visualization as final_result.png and the model as unet_voc_trained.pth

**How it works**
I started by creating the environment (conda)
Then, downloaded my dependencies such as PyTorch  with CUDA, matplolib, OpenCV, Ultralytics YOLO, using Anaconda prompt 
Run pretrained YOLOv8 nano on a sample image
Build the U-Net architecture
Train U-net for 5 epochs using Adam optimizer + CrossEntropyLoss
Last, I combined YOLO + U-Net into a single inference pipeline

**Files**

detection_segmentation.py is the main script
test_setup.py is a small file that verifies if my environment is properly responding
unet_voc_trained.pth is the trained U-Net weights (generated after step 5)
final_result.png is the 3-panel output generated after step 6
data/ is the Pascal VOC dataset ( DON'T try to download it via the python script, it will take half a day, instead download it externally and drop it in your main file script)

**How to run it**

python detection_segmentation.py 
It is important to know that it will take at least 10-15 minutes after launch to load. Mostly due to U-NET training.


**My setup**

Windows Laptop with NVIDIA GEFORCE RTX 3050 Ti with a small 4G VRAM...
Anaconda environment called cv-week4
Python latest version
Pytorch 2.5.1 with CUDA 12.1
