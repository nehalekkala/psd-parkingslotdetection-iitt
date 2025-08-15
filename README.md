Overview
Implements a deep learning system to detect parking slot occupancy in static images.

Uses VGG16-based transfer learning for binary classification.

Divides input image into a 20×16 grid (320 slots).

Classifies each slot as either Empty or Occupied.

Applies confidence thresholding (≥ 0.7) to improve accuracy.

Displays visual output with bounding boxes and confidence scores.

Features
VGG16 as the backbone model with frozen layers.

Grid-based slot segmentation and classification.

Two-class output: Empty / Occupied.

Confidence filtering for reliable predictions.

Annotated image output with slot status.

Tech Stack
Python 3.x

TensorFlow / Keras

OpenCV

NumPy

Matplotlib

How It Works
Data Preparation

Dataset split into empty and occupied categories.

Separate folders for training and testing.

Model Training

VGG16 used as a feature extractor.

First 10 layers frozen to retain pre-trained features.

Custom classification head added (Flatten → Dense).

Data augmentation applied for robustness.

Model trained for 10 epochs and saved as parking_status_model.h5.

Slot Detection

Input image divided into 320 grid cells (20 rows × 16 columns).

Each slot resized to 50×50 pixels and classified.

Confidence threshold applied to filter predictions.

Bounding boxes drawn with labels and confidence scores.

Output
Terminal displays:

Total number of slots.

Number of occupied and empty slots.

Processed image displayed with:

Red boxes for occupied slots.

Green boxes for empty slots.
