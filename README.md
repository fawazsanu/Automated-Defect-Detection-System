# Automated Industrial Defect Detection System 🏭👁️

## Project Overview
This project simulates an automated quality control pipeline for manufacturing. I engineered a Convolutional Neural Network (CNN) using **TensorFlow** to inspect components and classify them as either structurally sound or defective (e.g., surface cracks) in real-time.

To bypass data scarcity, I also built a dynamic data generation pipeline using OpenCV to manufacture a synthetic dataset of 2,000 factory parts.

**Key Features:**
- **Custom Synthetic Dataset:** OpenCV script that generates "metal" textures and randomizes simulated surface defects.
- **Deep Learning Model:** A TensorFlow-based CNN architecture optimized for binary classification of grayscale industrial images.
- **Real-Time Scanner:** An inference script that evaluates unseen images and overlays a QC pass/fail status directly onto the image feed.

## Tech Stack
- **Framework:** TensorFlow
- **Computer Vision:** OpenCV (`cv2`)
- **Data Manipulation:** NumPy

## How to Run the System

1. **Install Dependencies:**
   ```
   pip install -r requirements.txt

2. Generate the Synthetic Dataset (Optional):
   (Creates a factory_data folder with 2,000 images)
   ```
   python generate_factory_data.py

4. Train the Neural Network:
   ```
   python train_inspector.py

5. Run the Quality Control Scanner:
   (Tests a specific image against the trained .h5 model)
   ```
   python scan_part.py
