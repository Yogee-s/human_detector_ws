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
# CLASSES_TO_TRACK = ["person", "teleco"]
CLASSES_TO_TRACK = ["person"]

# camera → map transform
# 6 pair of points
T_MAP_CAM = np.array([
  [ 0.10099723, -0.36969855,  0.92364633, -3.98448572],
  [-0.99488402, -0.03537113,  0.09462916,  1.35687245],
  [-0.00231385, -0.92847825, -0.37137957,  1.35620029],
  [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
], dtype=float)


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
        
        # — start worker threads —
        threading.Thread(target=self.inference_loop, daemon=True).start()
        threading.Thread(target=self.publish_loop,   daemon=True).start()
    
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
            
            for box in res.boxes:
                # pull out coords
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                x1,x2 = int(x1*sx), int(x2*sx)
                y1,y2 = int(y1*sy), int(y2*sy)
                tid = int(box.id[0]) if USE_TRACKING and box.id is not None else None
                cls_idx = int(box.cls[0].cpu().numpy())
                cls_name = self.model.names[cls_idx]
                
                # depth at top-center of the box
                # cx, cy = (x1 + x2) // 2, y1
                # y0, y1_ = max(0, cy - 3), min(FRAME_H, cy + 4)
                # x0, x1_ = max(0, cx - 3), min(FRAME_W, cx + 4)
                # patch = depth[y0:y1_, x0:x1_]
                # if patch.size == 0:
                #     continue
                # z = float(np.median(patch)) * self.depth_scale
                # if z <= 0:
                #     continue

                # # depth at box center
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                y0, y1_ = max(0, cy - 3), min(FRAME_H, cy + 4)
                x0, x1_ = max(0, cx - 3), min(FRAME_W, cx + 4)
                patch = depth[y0:y1_, x0:x1_]
                if patch.size == 0:
                    continue
                z = float(np.median(patch)) * self.depth_scale
                if z <= 0:
                    continue

                # # depth at bottom‐center
                # cx,cy = (x1+x2)//2, y2
                # y0,y1_ = max(0,cy-3), min(FRAME_H,cy+4)
                # x0,x1_ = max(0,cx-3), min(FRAME_W,cx+4)
                # patch = depth[y0:y1_, x0:x1_]
                # if patch.size==0: 
                #     continue
                # z = float(np.median(patch))*self.depth_scale
                # if z<=0:
                #     continue
                
                # project to map
                Xc,Yc,Zc = rs.rs2_deproject_pixel_to_point(self.intr, [cx,cy], z)
                P = T_MAP_CAM @ np.array([Xc,Yc,Zc,1.0],float)
                Xm_,Ym_,_ = P[:3]
                
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