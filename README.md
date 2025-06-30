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

## 🎓 Model Training


### Training through Notebook

Open **train.ipynb** for an interactive workflow (visualize loss, mAP, etc).

### Training through CLI

```bash
python3 train.py \
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
python3 scripts/detection_scripts/simple_detector.py
```

---

## 🔧 Calibration & Publisher Scripts

### 1) **MQTT Calibration**

Compute the rigid‐body camera→map transform by collecting paired 3D points.

```bash
cd scripts/coordinate_conversion_scripts
python3 mqtt_calibration.py
```

* **\[SPACE]**: record one (camera, robot) correspondence
* **ESC or Ctrl+C**: finish and print a copy-pasteable `T_MAP_CAM = np.array([...], dtype=float)`

### 2) **MQTT Publisher**

Publish detections (`teleco` + `people`) every 50 ms:

```bash
python3 scripts/robot_publisher/mqtt_publisher.py
```
```

### 3) **Robot Subscriber**

Run on the robot to subscribe & create a topic in ROS 2:

```bash
python3 scripts/robot_publisher/mqtt_subscriber.py
```

Topics published:

* `/people_tracked` (PoseArray)
* `/human_markers`  (visualization\_msgs/Marker)

---

## 📡 UDP Alternative

If you need a non-MQTT fallback, use the UDP pair:

```bash
python3 scripts/robot_publisher/udp_publisher.py
python3 scripts/robot_publisher/udp_subscriber.py
```

---


## 📂 Repository Layout

```text
HUMAN_DETECTOR_WS/
├── data_raw/                  # raw RealSense & ROS-bag exports
├── data_training/             # train/val splits + data.yaml
├── models/                    # trained weights
├── scripts/
│   ├── coordinate_conversion_scripts/
│   │   ├── calibrate_transform.py
│   │   ├── coordinate_comparison.py
│   │   └── demo.py
│   ├── detection_scripts/
│   │   ├── custom_boundary.py    # Define detection boundary
│   │   ├── simple_detector.py    # Simple Detector script
│   │   ├── unique_id_tracker.py  # Calculate distance between chosen subjects
│   └── robot_publisher/
│       ├── mqtt_calibration.py   # collect (cam,map) pairs & print T_MAP_CAM
│       ├── mqtt_publisher.py     # publishes teleco+people → MQTT
│       ├── mqtt_subscriber.py    # runs on-robot: MQTT→ROS2 topics
│       ├── udp_publisher.py      # alternative UDP publisher
│       └── udp_subscriber.py     # alternative UDP → ROS2
├── src/                        # ROS2 detection & helper packages
└── README.md                   # this file
````

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
     pyyaml
   ```
3. **Build ROS workspace**:

   ```bash
   cd ~/Desktop/human_detector_ws
   source /opt/ros/humble/setup.bash
   colcon build --symlink-install
   source install/setup.bash
   ```

---



## 🔗 References

* **CVAT** annotation: [https://docs.cvat.ai/](https://docs.cvat.ai/)
* **Ultralytics YOLO**: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
* **YOLO-ROS wrapper**: [https://github.com/mgonzs13/yolo\_ros](https://github.com/mgonzs13/yolo_ros)

