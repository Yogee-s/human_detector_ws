# Human & Teleco Robot Detection
**Osaka University Frontier Programme: Social Robotics Group (Yoshikawa Lab)**

Real-time 2D detection of **humans** and a custom **Teleco robot**, plus 3D distance/position estimation using an Intel RealSense camera.

---

## 🔎 Project Highlights

- **Custom YOLOv11** trained on COCO-person + Teleco images
- **3D calibration** tool via MQTT to auto-compute camera→map transform
- **MQTT publisher** scripts for broadcasting detections
- **Inference** scripts (ROS 2 + mqtt) for on-robot integration

---

## 📋 Quick manual

### 1) MQTT Calibration
Compute the rigid-body camera→map transform by collecting paired 3D points.
```bash
python3 scripts/mqtt_publisher/mqtt_calibration.py
```
* **[SPACE]**: record one (camera, robot) correspondence
* **ESC or Ctrl+C**: finish and print a copy-pasteable `T_MAP_CAM = np.array([...], dtype=float)`

### 2) MQTT Publisher
Publish detections (`teleco` + `people`) every 50ms:
```bash
python3 scripts/mqtt_publisher/mqtt_publisher_with_boundary.py
```

### 3) MQTT Subscriber
Run on the robot to subscribe & create a topic in ROS 2:
```bash
python3 scripts/archive/mqtt_archive/mqtt_subscriber.py
```

Topics published:
* `/people_tracked` (PoseArray)
* `/human_markers` (visualization_msgs/Marker)
---

## 🎓 Model Training

### Training through Notebook
Open **yolo/data_training/train.ipynb** for an interactive workflow (visualize loss, mAP, etc).

### Training through CLI
```bash
python3 scripts/yolo/data_training/train.py \
  --data data_training/data.yaml \
  --model yolov11n.pt \
  --epochs 100 \
  --imgsz 640 \
  --batch 16
```

---

## 🚀 Running Inference

### A) 2D Only
```bash
ros2 launch human_detector human_detector_launch.py
```

### B) 2D + 3D Distance
```bash
ros2 launch yolo_ros yolov11_3d.launch.py
```

### C) Standalone Python Scripts
```bash
# Simple detection
python3 scripts/detection_scripts/simple_detector.py

# Custom boundary detection
python3 scripts/detection_scripts/custom_boundary.py

# Unique ID tracking with distance calculation
python3 scripts/detection_scripts/unique_id_tracker.py
```


## 📂 Repository Layout

```text
HUMAN_DETECTOR_WS/
├── build/                    # ROS2 build artifacts
├── install/                  # ROS2 install space
├── log/                      # ROS2 build logs
├── models/                   # trained YOLO weights
├── scripts/
│   ├── archive/              # archived/legacy scripts
│   ├── detection_scripts/
│   │   ├── custom_boundary.py      # Define detection boundary
│   │   ├── simple_detector.py      # Simple detector script
│   │   └── unique_id_tracker.py    # Distance tracking between subjects
│   ├── mqtt_publisher/
│   │   ├── helper_scripts/
│   │   ├── mqtt_config_python.yaml          # MQTT configuration
│   │   ├── mqtt_publisher_with_boundary.py  # Main MQTT publisher
│   │   └── mqtt_subscriber.py               # MQTT subscriber script for robot
│   ├── yolo/
│   │   ├── data_raw/            # Raw training data
│   │   └── data_training/       # Processed training datasets
├── src/                      # ROS2 packages
│   ├── human_detector/       # Main detection package
│   └── yolo_ros/            # YOLO ROS integration
├── .gitignore
└── README.md
```

---

## ⚙️ Dependencies & Installation

1. **ROS 2 Humble** (source install)
2. **Python 3.10+**:
   ```bash
   pip install \
     ultralytics \
     opencv-python \
     pyrealsense2 \
     paho-mqtt \
     pyyaml \
     numpy
   ```
3. **Build ROS workspace**:
   ```bash
   cd ~/Desktop/HUMAN_DETECTOR_WS
   source /opt/ros/humble/setup.bash
   colcon build --symlink-install
   source install/setup.bash
   ```

---

## 📡 Legacy/Archive Scripts

If you need alternative implementations, check the `scripts/archive/` folder for:
- UDP publisher/subscriber alternatives
- Legacy calibration scripts
- Older detection implementations

---

## 🔗 References

* **CVAT** annotation: [https://docs.cvat.ai/](https://docs.cvat.ai/)
* **Ultralytics YOLO**: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* **YOLO-ROS wrapper**: [https://github.com/mgonzs13/yolo_ros](https://github.com/mgonzs13/yolo_ros)

---