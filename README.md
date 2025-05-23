# Human + Teleco Robot Detection

A ROS 2 + YOLOv8 pipeline for real‐time 2D detection of **persons** and a custom **Teleco robot**, with 3D distance estimation via Intel RealSense depth data.

---

## 🚀 Features

- **2D Detection** of Person & Teleco robot
- **3D Distance Estimation** at each bounding‐box center
- **ROS 2 Node** publishes annotated images on `/human_detector/annotated`
- **Custom Training** on combined COCO‐person + Teleco datasets
- Jupyter notebooks for **data prep** & **model training**

---

## 📂 Repository Layout

```
HUMAN_DETECTOR_WS/
├── data_raw/                   # raw exports: ros bags, videos, CVAT/Roboflow outputs
│   ├── converted_mp4_videos/
│   └── labelled_data/
├── data_training/              # final train/val split for YOLO
│   ├── data_person/            # COCO‐person subset (train/val)
│   ├── data_teleco/            # Teleco robot images & labels
│   ├── images/
│   │   ├── train/
│   │   └── val/
│   ├── labels/
│   │   ├── train/
│   │   └── val/
│   └── data.yaml               # combined config (nc:2, names:['person','teleco'])
├── launch/
│   └── human_detector_launch.py
├── src/human_detector/
│   ├── human_detection_node.py
│   └── human_pose_node.py      # optional pose‐based variant
├── models/
│   └── best_*.pt               # your custom weights
├── helper_scripts.ipynb        # data prep & splitting notebooks
├── train.ipynb                 # example Ultralytics training notebook
├── package.xml & setup.py      # ROS 2 package metadata
└── README.md                   # ← this file
```

---

## ⚙️ Installation

1. **Clone & build the ROS 2 workspace**

   ```bash
   git clone https://github.com/YOUR_ORG/human_detector_ws.git
   cd human_detector_ws
   colcon build --merge-install
   source install/setup.bash
   ```

2. **Install Python deps** (for notebooks & scripts)

   ```bash
   pip install ultralytics opencv-python pyrealsense2 pyyaml
   ```

---

## 📊 Data Preparation

1. **Export** frames/video from ROS-bag → annotate in CVAT/Roboflow.
2. **Run** `helper_scripts.ipynb` to:

   - Collect Teleco & COCO-person exports
   - Split into `train/val` under `data_training/`
   - Generate `data_training/data.yaml`:

     ```yaml
     train: images/train
     val: images/val
     nc: 2
     names:
       - person
       - teleco
     ```

3. **Verify** `data_training/images/{train,val}` and `labels/{train,val}` exist.

---

## 🤖 Model Training

### CLI

```bash
yolo train \
  data=data_training/data.yaml \
  model=yolov8n.pt \
  epochs=120 \
  imgsz=416 \
  batch=16 \
  project=runs/train \
  name=human_teleco
```

### Jupyter

In **train.ipynb**:

```python
from ultralytics import YOLO

model = YOLO('models/yolov8n.pt')  # base checkpoint
results = model.train(
    data='data_training/data.yaml',
    epochs=120,
    imgsz=416,
    batch=16,
    device=0
)
print("Best weights saved at:", results.best)
```

---

## 🚨 Inference with ROS 2

Launch live detection + distance estimation:

```bash
source install/setup.bash
ros2 launch human_detector human_detector_launch.py
```

- **Input:**

  - `/camera/realsense2_camera/color/image_raw`
  - `/camera/realsense2_camera/depth/image_rect_raw`

- **Output:**

  - `/human_detector/annotated` (`sensor_msgs/Image`)

To swap models, edit the `model_path` parameter in `launch/human_detector_launch.py` to your `best_*.pt`.

---

## 🖥 Viewing Results

- An OpenCV window shows 2D boxes + distance labels.
- Alternatively, view `/human_detector/annotated` in **RViz** or **rqt_image_view**.

---

## 🛠 Customization

- Use `human_detection_node.py` for pure detection (no keypoints).
- Use `human_pose_node.py` if you want pose‐keypoints.
- Adjust YOLO training parameters (augmentation, freeze layers, etc.) in `train.ipynb`.

---

## 📚 Notebooks

- **helper_scripts.ipynb** — frame extraction, dataset merge/split, `data.yaml` generation
- **train.ipynb** — interactive training & metric plotting
- **inference.ipynb** — test detection on images/video

---

## 🤝 Contributing

Contributions welcome! Please open issues or PRs for enhancements.
Released under the **MIT License**.

---

© 2025 Yogee-s — Bridging robotics with state-of-the-art vision.
