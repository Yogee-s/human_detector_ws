#!/usr/bin/env python3
import threading
import time
import json

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import torch

from helper_scripts.mqtt_client import MQTT

torch.backends.cudnn.benchmark = True

# === CONFIGURATION ===
POSE_MODEL_PATH = '/home/commu/Desktop/human_detector_ws/models/yolov11n-pose.pt'  # Pose model for person detection and height
FRAME_W,FRAME_H = 640,480
USE_TRACKING  = True
TRACKER_CFG   = 'bytetrack.yaml'

# YOLO tracking
CONF_THRESH   = 0.6
IOU_THRESH    = 0.8

DET_W,DET_H   = 320,240
ALPHA_MAP     = 0.85 # EMA smoothing factor for map coordinates

# === CLASS FILTERING CONFIGURATION ===
# Only tracking persons since pose model handles both detection and height
CLASSES_TO_TRACK = ["person"]

# camera → map transform
T_MAP_CAM = np.array([
  [ 0.10099723, -0.36969855,  0.92364633, -3.98448572],
  [-0.99488402, -0.03537113,  0.09462916,  1.35687245],
  [-0.00231385, -0.92847825, -0.37137957,  1.35620029],
  [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
], dtype=float)


class PoseHeightDetector:
    def __init__(self):
        # COCO pose keypoints (17 keypoints for human pose)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Indices for key points
        self.NOSE_IDX = 0
        self.LEFT_ANKLE_IDX = 15
        self.RIGHT_ANKLE_IDX = 16
        self.LEFT_KNEE_IDX = 13
        self.RIGHT_KNEE_IDX = 14
        
    def get_ground_level(self, keypoints):
        """
        Estimate ground level from ankle positions
        """
        if keypoints.shape[0] <= max(self.LEFT_ANKLE_IDX, self.RIGHT_ANKLE_IDX):
            return None
            
        left_ankle = keypoints[self.LEFT_ANKLE_IDX]
        right_ankle = keypoints[self.RIGHT_ANKLE_IDX]
        
        ground_candidates = []
        
        if len(left_ankle) >= 3 and left_ankle[2] > 0.5:
            ground_candidates.append(left_ankle[1])
        if len(right_ankle) >= 3 and right_ankle[2] > 0.5:
            ground_candidates.append(right_ankle[1])
            
        if ground_candidates:
            return max(ground_candidates)
        else:
            return None
    
    def calculate_nose_height_3d(self, keypoints, depth_image, depth_intrinsics, depth_scale):
        """
        Calculate real-world height from ground to nose using depth data
        """
        if keypoints.shape[0] <= self.NOSE_IDX:
            return None
            
        nose = keypoints[self.NOSE_IDX]
        
        if len(nose) < 3 or nose[2] < 0.5:
            return None
            
        # Get 3D coordinates for nose
        nose_x, nose_y = int(nose[0]), int(nose[1])
        nose_depth = self.get_depth_at_point(depth_image, nose_x, nose_y) * depth_scale
        if nose_depth <= 0:
            return None
            
        nose_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [nose_x, nose_y], nose_depth)
        
        # Get 3D coordinates for ground reference (ankles)
        ground_3d_points = []
        
        if (keypoints.shape[0] > self.LEFT_ANKLE_IDX and 
            len(keypoints[self.LEFT_ANKLE_IDX]) >= 3 and 
            keypoints[self.LEFT_ANKLE_IDX][2] > 0.5):
            left_ankle = keypoints[self.LEFT_ANKLE_IDX]
            ankle_x, ankle_y = int(left_ankle[0]), int(left_ankle[1])
            ankle_depth = self.get_depth_at_point(depth_image, ankle_x, ankle_y) * depth_scale
            if ankle_depth > 0:
                ankle_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [ankle_x, ankle_y], ankle_depth)
                ground_3d_points.append(ankle_3d)
        
        if (keypoints.shape[0] > self.RIGHT_ANKLE_IDX and 
            len(keypoints[self.RIGHT_ANKLE_IDX]) >= 3 and 
            keypoints[self.RIGHT_ANKLE_IDX][2] > 0.5):
            right_ankle = keypoints[self.RIGHT_ANKLE_IDX]
            ankle_x, ankle_y = int(right_ankle[0]), int(right_ankle[1])
            ankle_depth = self.get_depth_at_point(depth_image, ankle_x, ankle_y) * depth_scale
            if ankle_depth > 0:
                ankle_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [ankle_x, ankle_y], ankle_depth)
                ground_3d_points.append(ankle_3d)
        
        if not ground_3d_points:
            return None
            
        # Use average Y coordinate of ankles as ground level
        avg_ground_y = sum(point[1] for point in ground_3d_points) / len(ground_3d_points)
        
        # Calculate height (nose Y - ground Y, but Y is negative upward in camera frame)
        height_3d = avg_ground_y - nose_3d[1]
        
        return height_3d if height_3d > 0 else None
    
    def get_depth_at_point(self, depth_image, x, y, patch_size=5):
        """
        Get median depth value at a point using a small patch
        """
        half = patch_size // 2
        h, w = depth_image.shape
        
        y_min, y_max = max(0, y - half), min(h, y + half + 1)
        x_min, x_max = max(0, x - half), min(w, x + half + 1)
        
        patch = depth_image[y_min:y_max, x_min:x_max]
        if patch.size == 0:
            return 0
        return np.median(patch)


def draw_text_block(img, text_lines, position, font=cv2.FONT_HERSHEY_SIMPLEX, 
                   font_scale=0.45, text_color=(255, 255, 255), 
                   bg_color=(0, 0, 0, 180), thickness=1, padding=6, line_spacing=16):
    """
    Draw multiple lines of text with a single unified background
    """
    x, y = position
    
    if not text_lines:
        return
    
    # Calculate dimensions for all text lines
    max_width = 0
    total_height = 0
    line_heights = []
    
    for text in text_lines:
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        max_width = max(max_width, text_width)
        line_heights.append(text_height)
        total_height += line_spacing
    
    # Remove extra spacing from last line
    total_height -= (line_spacing - max(line_heights))
    
    # Create unified background rectangle
    bg_x1 = x - padding
    bg_y1 = y - padding
    bg_x2 = x + max_width + padding
    bg_y2 = y + total_height + padding
    
    # Create overlay for semi-transparent background
    overlay = img.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color[:3], -1)
    
    # Apply the overlay with transparency
    alpha = bg_color[3] / 255.0 if len(bg_color) > 3 else 0.7
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    
    # Draw all text lines
    current_y = y + line_heights[0] if line_heights else y
    for i, text in enumerate(text_lines):
        cv2.putText(img, text, (x, current_y), font, font_scale, text_color, thickness)
        current_y += line_spacing


class HumanPublisher:
    def __init__(self):
        # — start MQTT client —
        self.mqtt = MQTT()
        threading.Thread(target=self.mqtt.connect, daemon=True).start()
        time.sleep(1.0)
        
        # — load pose model (handles both detection and height) —
        self.model = YOLO(POSE_MODEL_PATH)
        self.model.fuse()
        self.model = self.model.to('cuda').half()
        self.pose_detector = PoseHeightDetector()
        
        # Pose model only detects person class (class 0)
        self.class_indices = [0]  # person class
        
        print(f"Using pose model for person detection and height estimation")
        
        # — RealSense setup —
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, 30)
        profile = self.pipeline.start(cfg)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.intr = profile.get_stream(rs.stream.depth)\
                             .as_video_stream_profile()\
                             .get_intrinsics()
        
        # — shared state —
        self.running = True
        self.lock = threading.Lock()
        self.latest_dets = []   # list of (x, y, height)
        self.map_ema = {}       # id → EMA position
        
        # — start worker threads —
        threading.Thread(target=self.inference_loop, daemon=True).start()
        threading.Thread(target=self.publish_loop, daemon=True).start()
    
    def calculate_height_from_keypoints(self, keypoints, depth):
        """
        Calculate height directly from pose keypoints
        """
        try:
            height = self.pose_detector.calculate_nose_height_3d(
                keypoints, depth, self.intr, self.depth_scale
            )
            return height
        except Exception as e:
            print(f"Error in height calculation: {e}")
            return None
    
    def inference_loop(self):
        sx, sy = FRAME_W/DET_W, FRAME_H/DET_H
        cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
        
        while self.running:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df:
                continue
            
            img = np.asanyarray(cf.get_data())
            depth = cv2.medianBlur(np.asanyarray(df.get_data()), 5)
            
            # Run pose detection and tracking directly on full resolution image
            res = self.model.track(
                img,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                tracker=TRACKER_CFG,
                persist=True,
                classes=self.class_indices
            )[0]
            
            dets = []
            vis = img.copy()
            
            # Process pose detections
            if res.keypoints is not None and len(res.keypoints.data) > 0:
                keypoints_data = res.keypoints.data.cpu().numpy()
                boxes = res.boxes.data.cpu().numpy() if res.boxes is not None else None
                
                for i, keypoints in enumerate(keypoints_data):
                    # Get box information
                    if boxes is not None and len(boxes) > i:
                        x1, y1, x2, y2 = map(int, boxes[i][:4])
                        confidence = float(boxes[i][4])
                        tid = int(boxes[i][5]) if USE_TRACKING and len(boxes[i]) > 5 else None
                        
                        # Calculate center point for depth
                        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                        y0, y1_ = max(0, cy - 3), min(FRAME_H, cy + 4)
                        x0, x1_ = max(0, cx - 3), min(FRAME_W, cx + 4)
                        patch = depth[y0:y1_, x0:x1_]
                        if patch.size == 0:
                            continue
                        z = float(np.median(patch)) * self.depth_scale
                        if z <= 0:
                            continue
                        
                        # Calculate height from keypoints
                        height = self.calculate_height_from_keypoints(keypoints, depth)
                        
                        # Project to map coordinates
                        Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(self.intr, [cx, cy], z)
                        P = T_MAP_CAM @ np.array([Xc, Yc, Zc, 1.0], float)
                        Xm_, Ym_, _ = P[:3]
                        
                        # Apply EMA smoothing if tracking
                        if tid is not None:
                            prev = self.map_ema.get(tid, np.array([Xm_, Ym_], float))
                            filt = ALPHA_MAP * prev + (1 - ALPHA_MAP) * np.array([Xm_, Ym_], float)
                            self.map_ema[tid] = filt
                            Xm, Ym = filt
                        else:
                            Xm, Ym = Xm_, Ym_
                        
                        dets.append((float(Xm), float(Ym), height))
                        
                        # Visualization with enhanced appearance
                        box_color = (147, 20, 255)  # Pink color (BGR format)
                        cv2.rectangle(vis, (x1, y1), (x2, y2), box_color, 2)
                        
                        # Draw keypoints
                        for j, (kx, ky, kconf) in enumerate(keypoints):
                            if kconf > 0.5:
                                # Nose keypoint in red, others in green
                                kp_color = (0, 0, 255) if j == self.pose_detector.NOSE_IDX else (0, 255, 0)
                                cv2.circle(vis, (int(kx), int(ky)), 3, kp_color, -1)
                        
                        # Create organized text with proper labels
                        text_lines = [
                            f"Pos: ({Xm:.2f}, {Ym:.2f})",
                            f"Dist: {z:.2f}m"
                        ]
                        
                        # Add height info if available
                        if height is not None:
                            text_lines.append(f"Height: {height:.2f}m")
                        else:
                            text_lines.append("Height: N/A")
                        
                        # Draw unified text block inside the box
                        text_x_offset = x1 + 8   # Small margin from left edge
                        text_y_offset = y1 + 8   # Small margin from top edge
                        
                        draw_text_block(vis, text_lines, (text_x_offset, text_y_offset), 
                                      font_scale=0.45, text_color=(255, 255, 255), 
                                      bg_color=(0, 0, 0, 180), thickness=1)
            
            # Update shared detections
            with self.lock:
                self.latest_dets = dets
            
            # Show window
            cv2.imshow("Detections", vis)
            if cv2.waitKey(1) == 27:  # ESC to quit
                self.stop()
                break
    
    def publish_loop(self):
        while self.running:
            self.send_people_mqtt()
            time.sleep(0.2) # publish every 200ms (5Hz)
    
    def send_people_mqtt(self):
        with self.lock:
            dets = list(self.latest_dets)
        
        # Create people list with height data
        people = []
        for x, y, height in dets:
            person_data = {"x": x, "y": y, "z": 0.0}
            if height is not None:
                person_data["height"] = height
            people.append(person_data)
        
        msg = {
            "timestamp": time.time(),
            "frame_id": "map",
            "people": people
        }
        try:
            self.mqtt.publish_human_results(json.dumps(msg))
            print("[mqtt] published:", msg)
        except Exception as e:
            print("[mqtt error]", e)
    
    def stop(self):
        self.running = False
        self.pipeline.stop()
        cv2.destroyAllWindows()

def main():
    pub = HumanPublisher()
    try:
        while pub.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pub.stop()

if __name__=='__main__':
    main()