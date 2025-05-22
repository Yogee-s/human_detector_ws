# YOLOv8 Training Script
from ultralytics import YOLO

model = YOLO('../src/human_detector/models/yolov11n.pt') # Load a pretrained YOLOv8n model

# Training
#    - data: path to your data.yaml
#    - epochs, imgsz, project, name: same CLI args
results = model.train(
    data    = 'training_data/data.yaml',
    epochs  = 50,
    imgsz   = 640,
    project = 'human_robot',
    name    = 'exp1'
)

# `results` is a Python object you can inspect for metrics, best.pt path, etc.
print(f"Trained checkpoint saved to: {results.best}")
