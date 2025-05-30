# YOLOv11 Training Script with Full Parameter Explanations and Improved Stability

from ultralytics import YOLO
import os
from datetime import datetime

# --- Model Setup ---
# Load a pretrained YOLOv8n model (nano version: fastest, smallest, good for limited data or edge devices)
model = YOLO('../src/human_detector/models/yolov11s.pt')
# To resume training from a previous checkpoint:
# model = YOLO('/path/to/your/best.pt')

# --- Training Configuration ---
epochs = 300               # Number of times the model sees the entire dataset
imgsz = 640                # Size to which all images are resized (square). 640 is a good trade-off between speed and accuracy
batch_size = 4             # Number of images used in one forward/backward pass (small for limited VRAM)

lr0 = 1e-5                 # Initial learning rate (controls how much the model updates each step)
lrf = 0.01                 # Final learning rate = lr0 * lrf, used with cosine decay learning schedule
optimizer = "Adam"          # Optimizer algorithm: 'SGD' is more stable for small datasets, 'AdamW' (default) adapts better to large, noisy datasets
# warmup_epochs = 3          # Number of epochs to slowly ramp up learning rate to avoid instability at start
dropout = 0.1              # Dropout rate (randomly turns off neurons during training to prevent overfitting)

# --- Experiment Folder Naming ---
time_stamp = datetime.now().strftime("%Y%m%d_%H%M")

exp_name = (
    f"yolov11s_v1_"         # Model name and version. You can use this to identify custom model variants.
    f"ep{epochs}_"          # Number of epochs (e.g., ep300 means trained for 300 full dataset passes)
    f"img{imgsz}_"          # Image input size (e.g., img640 means all images resized to 640x640)
    f"bs{batch_size}_"      # Batch size (e.g., bs4 means 4 images processed per training step)
    f"lr{lr0}_"             # Initial learning rate (e.g., lr0.0001)
    f"lrf{lrf}_"            # Final learning rate factor (e.g., lrf0.01 → lr drops to 0.000001 at end)
    f"{optimizer}_"         # Optimizer used (e.g., SGD, AdamW, etc.)
    f"augmented_"            # Tag for augmentation setup (e.g., Final or custom profile name)
    f"{time_stamp}"         # Timestamp for uniqueness and chronological sorting
)

# --- Training Start ---
results = model.train(
    seed=42,                     # Random seed for reproducibility
    data='data.yaml',            # Path to data config file (defines class names and image paths)
    epochs=epochs,               # Total number of training epochs
    patience=50,                 # Early stopping: if val loss doesn't improve for 10 epochs, stop training
    dropout=dropout,             # Helps prevent overfitting by randomly disabling neurons
    imgsz=imgsz,                 # Resizes all images to this square dimension
    batch=batch_size,            # Number of images per training batch
    lr0=lr0,                     # Starting learning rate
    lrf=lrf,                     # Final learning rate multiplier
    # warmup_epochs=warmup_epochs,# Smooth ramp-up of LR to avoid unstable gradients
    optimizer=optimizer,         # Optimizer choice: SGD recommended for small datasets
    device=0,                    # GPU device ID (0 = first GPU)
    half=True,                   # Use float16 precision if supported (faster training)
    workers=2,                   # Number of CPU workers for data loading (adjust based on CPU)
    cache=True,                  # Preload and cache images in memory for faster epoch execution
    # rect=True,                   # Enables rectangular training for better image aspect ratio handling
    verbose=True,                # Print detailed training logs for every batch
    amp=False ,
    # exist_ok=True,               # Overwrites existing output folder if it already exists

    # --- Data Augmentation (tuned for small datasets) ---
    hsv_h=0.015,                 # Hue variation (color tint)
    hsv_s=0.7,                   # Saturation variation (color strength)
    hsv_v=0.4,                   # Brightness variation
    translate=0.05,              # Random image shift (reduced for safety)
    scale=0.1,                   # Random zoom in/out (less aggressive)
    fliplr=0.5,                  # 50% chance to flip images horizontally
    mosaic=0.2,                  # Combines 4 training images into one
    mixup=0.03,                  # Blends two images and labels
    copy_paste=0.03,             # Copies object from one image and pastes onto another
    perspective=0.001,           # Small perspective distortion to improve robustness

    # --- Output Settings ---
    project='results',           # Base directory to store training logs and weights
    name=exp_name                # Subfolder name under project/ to store this run’s output
)

# --- Save Path Display ---
save_dir = str(results.save_dir)
best_pt = os.path.join(save_dir, 'weights', 'best.pt')
print(f"✅ Trained checkpoint saved to: {best_pt}")
