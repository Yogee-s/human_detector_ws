# Human & Teleco Robot Detection

**Osaka University Frontier Programme: Social Robotics Group (Yoshikawa Lab)**
---
Real‐time 2D detection of **humans** and a custom **Teleco robot**, plus 3D distance/position estimation using an Intel RealSense camera.

---

## 🔎 Project Overview

* **Context:** Developed at Osaka Univ. Frontier Programme (Yoshikawa Lab) for human/Teleco detection.
* **Hardware:** Intel RealSense D435 for RGB + depth streams.
* **Model:** Ultralytics YOLO (`yolo11n.pt`) custom‐trained on COCO‐person + Teleco datasets.
* **ROS Integration:** Wraps detection into ROS 2 nodes; optionally publishes 3D point clouds.

---

## 📂 Repository Layout

```text
HUMAN_DETECTOR_WS/
├── data_raw/            # raw exports from RealSense & ROS bags
│   ├── converted_mp4_videos/
│   └── labelled_data/   # CVAT/Roboflow annotations
├── data_training/       # train/val splits & YAML config
│   ├── data_person/     # COCO‑person subset
│   ├── data_teleco/     # Teleco frames & labels
│   └── data.yaml        # nc:2, names: ['person','teleco']
├── detection_scripts/   # standalone Python demos
│   ├── laptop_human_publisher.py
│   └── simple_detector.py
├── src/
│   ├── human_detector/  # ROS 2 detection package
│   ├── yolo_ros/        # upstream ROS wrapper (mgonzs13/
└── README.md            # this file
```

---

## ⚙️ Dependencies & Installation

1. **ROS 2 Humble** (source install)
2. **Python 3.10+** with:

   ```bash
   pip install ultralytics opencv-python pyrealsense2 pyyaml
   ```
3. **Build workspace**

   ```bash
   cd ~/Desktop/human_detector_ws
   source /opt/ros/humble/setup.bash
   colcon build --symlink-install
   source install/setup.bash
   ```

---

## 🛠 Data Processing & Annotation

1. **Record raw data** via RealSense / ROS‐bag.
2. **Convert & extract frames** using `data_raw/download_human_data.py` or `helper_scripts.ipynb`.
3. **Label** in CVAT (see [https://docs.cvat.ai/docs/administration/basics/installation/](https://docs.cvat.ai/docs/administration/basics/installation/)).
4. **Prepare** `data_training/` splits and `data.yaml` with helper notebooks.

---

## 🎓 Model Training

### Headless (CLI)

```bash
python3 train.py \
  --data data_training/data.yaml \
  --model yolov11n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16
```

### Interactive (Notebook)

Open **train.ipynb** and follow cells to train & visualize metrics.

---

## 🚀 Inference & ROS 2 Usage

### 1) 2D Detection only

```bash
ros2 launch human_detector human_detector_launch.py
```

### 2) 2D + 3D Distance (with yolo\_ros)

```bash
ros2 launch yolo_bringup yolov11.launch.py use_3d:=True
```

### 3) Standalone Python demos

```bash
python3 detection_scripts/simple_detector.py
python3 detection_scripts/laptop_human_publisher.py
```

---

## 🔗 Upstream & References

* **CVAT** for annotation: [https://docs.cvat.ai/docs/administration/basics/installation/](https://docs.cvat.ai/docs/administration/basics/installation/)
* **YOLO‐ROS** wrapper: [https://github.com/mgonzs13/yolo\_ros](https://github.com/mgonzs13/yolo_ros)
* **COCO** open-source person dataset: [https://docs.ultralytics.com/datasets/detect/coco/](https://docs.ultralytics.com/datasets/detect/coco/)

---

