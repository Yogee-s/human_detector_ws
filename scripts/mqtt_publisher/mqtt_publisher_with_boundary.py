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
MODEL_PATH    = '/home/commu/Desktop/human_detector_ws/models/best_yolo11m.pt'
FRAME_W,FRAME_H = 640,480
USE_TRACKING  = True
TRACKER_CFG   = 'bytetrack.yaml'

# YOLO tracking
#   conf      = 0.6    # Detection confidence threshold: 
#                      #   ↑ higher → fewer false positives, but may miss small/occluded objects
#                      #   ↓ lower  → more detections, but more noise
#
#   iou       = 0.7    # NMS IoU threshold:
#                      #   ↓ lower → stricter merging, less box overlap
#                      #   ↑ higher→ allow closer boxes, may keep duplicates
CONF_THRESH   = 0.4
IOU_THRESH    = 0.8 #0.5

DET_W,DET_H   = 320,240
ALPHA_MAP     = 0.85 # EMA smoothing factor for map coordinates

# === CLASS FILTERING CONFIGURATION ===
CLASSES_TO_TRACK = ["person", "teleco"]
# CLASSES_TO_TRACK = ["person"]

# === BOUNDARY CONFIGURATION ===
# Define your 4-point boundary in map coordinates (x, y)
# Replace these coordinates with your actual boundary points from SLAM map
BOUNDARY_POINTS_MAP = np.array([
    [-3.101, 0.804],  # Point 1
    [-1.680, 0.723],  # Point 2
    [-1.491, -1.203],  # Point 3
    [-3.150, -0.049],  # Point 4
], dtype=np.float32)

# camera → map transform
# 6 pair of points
T_MAP_CAM = np.array([
  [ 0.09589393, -0.45845295,  0.88352999, -4.00730774],
  [-0.98829735,  0.06193290,  0.13940109,  0.54129092],
  [-0.11862842, -0.88655807, -0.44714885,  1.63494930],
  [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
], dtype=float)


def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting algorithm"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside


def map_to_pixel(map_point, transform_matrix, intrinsics):
    """Convert map coordinates to pixel coordinates for visualization"""
    # Convert map to camera coordinates (inverse transform)
    try:
        T_CAM_MAP = np.linalg.inv(transform_matrix)
        map_homogeneous = np.array([map_point[0], map_point[1], 0.0, 1.0])
        cam_coords = T_CAM_MAP @ map_homogeneous
        
        # Project to pixel coordinates
        if cam_coords[2] > 0:  # Only if in front of camera
            pixel = rs.rs2_project_point_to_pixel(intrinsics, cam_coords[:3])
            return int(pixel[0]), int(pixel[1])
    except:
        pass
    return None


class HumanPublisher:
    def __init__(self):
        # — start MQTT client —
        self.mqtt = MQTT()
        threading.Thread(target=self.mqtt.connect, daemon=True).start()
        time.sleep(1.0)  # give MQTT a moment
        
        # — load YOLO model —
        self.model = YOLO(MODEL_PATH)
        self.model.fuse()
        self.model = self.model.to('cuda').half()
        
        # Convert class names to indices for filtering
        self.class_indices = []
        if isinstance(CLASSES_TO_TRACK[0], str):
            # Convert class names to indices
            for class_name in CLASSES_TO_TRACK:
                for idx, name in self.model.names.items():
                    if name == class_name:
                        self.class_indices.append(idx)
                        break
                else:
                    print(f"Warning: Class '{class_name}' not found in model")
        else:
            # Assume they're already indices
            self.class_indices = CLASSES_TO_TRACK
        
        print(f"Tracking classes: {[self.model.names[i] for i in self.class_indices]}")
        
        # — RealSense setup —
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, 30)
        profile = self.pipeline.start(cfg)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.intr  = profile.get_stream(rs.stream.depth)\
                             .as_video_stream_profile()\
                             .get_intrinsics()
        
        # — shared state —
        self.running     = True
        self.lock        = threading.Lock()
        self.latest_dets = []   # list of (x, y, cls_name)
        self.map_ema     = {}   # id → EMA position
        
        # — boundary setup —
        self.boundary_points = BOUNDARY_POINTS_MAP
        print(f"Boundary points (map coordinates): {self.boundary_points}")
        
        # — start worker threads —
        threading.Thread(target=self.inference_loop, daemon=True).start()
        threading.Thread(target=self.publish_loop,   daemon=True).start()
    
    def draw_boundary(self, img):
        """Draw the boundary polygon on the image"""
        # Convert boundary points from map to pixel coordinates
        pixel_points = []
        for point in self.boundary_points:
            pixel_coord = map_to_pixel(point, T_MAP_CAM, self.intr)
            if pixel_coord is not None:
                pixel_points.append(pixel_coord)
        
        if len(pixel_points) >= 3:  # Need at least 3 points to draw a polygon
            # Draw faint boundary lines
            pts = np.array(pixel_points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            
            # Draw polygon with faint green color (semi-transparent effect)
            overlay = img.copy()
            cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)
            cv2.fillPoly(overlay, [pts], (0, 255, 0))
            
            # Blend with original image for transparency effect
            cv2.addWeighted(overlay, 0.1, img, 0.9, 0, img)
            
            # Draw boundary lines more prominently
            cv2.polylines(img, [pts], True, (0, 255, 0), 2)
    
    def inference_loop(self):
        sx, sy = FRAME_W/DET_W, FRAME_H/DET_H
        cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
        
        while self.running:
            frames  = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df:
                continue
            
            img   = np.asanyarray(cf.get_data())
            depth = cv2.medianBlur(np.asanyarray(df.get_data()), 5)
            small = cv2.resize(img, (DET_W, DET_H))
            
            # Run inference with class filtering
            res = self.model.track(
                small,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                tracker=TRACKER_CFG,
                persist=True,
                classes=self.class_indices  # Only detect specified classes
            )[0]
            
            dets = []
            # draw on copy for visualization
            vis = img.copy()
            
            # Draw boundary first (so it appears behind detections)
            self.draw_boundary(vis)
            
            for box in res.boxes:
                # pull out coords
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                x1,x2 = int(x1*sx), int(x2*sx)
                y1,y2 = int(y1*sy), int(y2*sy)
                tid = int(box.id[0]) if USE_TRACKING and box.id is not None else None
                cls_idx = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names[cls_idx]
                
                # depth at box center
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                y0, y1_ = max(0, cy - 3), min(FRAME_H, cy + 4)
                x0, x1_ = max(0, cx - 3), min(FRAME_W, cx + 4)
                patch = depth[y0:y1_, x0:x1_]
                if patch.size == 0:
                    continue
                z = float(np.median(patch)) * self.depth_scale
                if z <= 0:
                    continue
                
                # project to map
                Xc,Yc,Zc = rs.rs2_deproject_pixel_to_point(self.intr, [cx,cy], z)
                P = T_MAP_CAM @ np.array([Xc,Yc,Zc,1.0],float)
                Xm_,Ym_,_ = P[:3]
                
                # Check if the detection is within the boundary
                if not point_in_polygon([Xm_, Ym_], self.boundary_points):
                    # If you want to visualize detections outside the boundary,
                    # # Draw grayed out box for detections outside boundary
                    # cv2.rectangle(vis, (x1,y1), (x2,y2), (128, 128, 128), 2)
                    # text = f"{cls_name} (OUT)"
                    # cv2.putText(vis, text, (x1, y1-10),
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
                    continue  # Skip this detection
                
                if tid is not None:
                    prev = self.map_ema.get(tid, np.array([Xm_,Ym_],float))
                    filt = ALPHA_MAP*prev + (1-ALPHA_MAP)*np.array([Xm_,Ym_],float)
                    self.map_ema[tid]=filt
                    Xm,Ym = filt
                else:
                    Xm,Ym = Xm_,Ym_
                
                dets.append((float(Xm), float(Ym), cls_name))
                
                # draw box + label + coords with different colors for different classes
                if cls_name == "person":
                    color = (255, 0, 0)  # Blue for person
                elif cls_name == "teleco":
                    color = (0, 0, 255)  # Red for teleco
                else:
                    color = (0, 255, 0)  # Green for other classes if they are needed
                
                cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
                text = f"{cls_name} ({Xm:.2f},{Ym:.2f})"
                cv2.putText(vis, text, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # update shared detections
            with self.lock:
                self.latest_dets = dets
            
            # Add boundary info to window
            cv2.putText(vis, f"Boundary: {len(self.boundary_points)} points", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # show window
            cv2.imshow("Detections", vis)
            if cv2.waitKey(1)==27:  # ESC to quit
                self.stop()
                break
    
    def publish_loop(self):
        while self.running:
            self.send_people_mqtt()
            time.sleep(0.2) # publish every 200ms (5Hz)
            # time.sleep(0.05) # publish every 50ms (20Hz)
    
    def send_people_mqtt(self):
        with self.lock:
            dets = list(self.latest_dets)
        
        # separate lists
        teleco = None
        people = []
        for x,y,cls_name in dets:
            if cls_name == "teleco":
                teleco = {"x":x,"y":y,"z":0.0}
            elif cls_name == "person":
                people.append({"x":x,"y":y,"z":0.0})
            # Add other classes to people list if needed
            # else:
            #     people.append({"x":x,"y":y,"z":0.0})
        
        msg = {
            "timestamp": time.time(),
            "frame_id":  "map",
            "teleco":    teleco,
            "people":    people
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