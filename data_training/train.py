from ultralytics import YOLO
import os
from datetime import datetime

# --- Model Setup ---
model = YOLO('../src/human_detector/models/yolov11s.pt')  # Or path to your checkpoint

# --- Hyperparameters ---
epochs = 200
imgsz = 640                     # Larger image size to push GPU harder
batch_size = 16                 # Increase batch size to use more VRAM
lr0 = 0.01
lrf = 0.01
optimizer = "SGD"
warmup_epochs = 3
dropout = 0.1
patience = 20


# --- Experiment Folder Naming ---
time_stamp = datetime.now().strftime("%Y%m%d_%H%M")
exp_name = (
    f"yolov11s_"
    f"ep{epochs}_"
    f"img{imgsz}_"
    f"bs{batch_size}_"
    f"lr{lr0}_"
    f"{optimizer}_"
    f"{time_stamp}"
)

# --- Training Start ---
results = model.train(
    seed=42,
    data='data.yaml',            # Your data.yaml file with class names
    epochs=epochs,
    patience=patience,
    dropout=dropout,
    imgsz=imgsz,
    batch=batch_size,
    lr0=lr0,
    lrf=lrf,
    warmup_epochs=warmup_epochs,
    optimizer=optimizer,
    device=0,                    # Use first GPU
    amp=True,                    # Use automatic mixed precision (faster)
    half=False,                  # Not needed with amp=True
    workers=6,                   # Adjust based on your CPU (e.g., 6 cores)
    cache=True,                  # Cache dataset in memory (if RAM allows)
    rect=False,                  # Use full augmentation (important!)
    verbose=True,

    # --- Data Augmentation (reduced mixup/mosaic for speed) ---
    hsv_h=0.01,
    hsv_s=0.4,
    hsv_v=0.3,
    translate=0.03,
    scale=0.05,
    fliplr=0.5,
    mosaic=0.05,
    mixup=0.0,
    copy_paste=0.0,
    perspective=0.001,

    # --- Output Directory ---
    project='results',
    name=exp_name
)

# --- Save Path Display ---
save_dir = str(results.save_dir)
best_pt = os.path.join(save_dir, 'weights', 'best.pt')
print(f"\n✅ Trained checkpoint saved to: {best_pt}")
