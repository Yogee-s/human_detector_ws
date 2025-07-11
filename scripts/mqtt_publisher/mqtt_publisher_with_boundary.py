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
FRAME_W, FRAME_H = 640, 480
USE_TRACKING  = True
TRACKER_CFG   = 'bytetrack.yaml'

CONF_THRESH   = 0.4
IOU_THRESH    = 0.8

DET_W, DET_H  = 320, 240
ALPHA_MAP     = 0.85

# === CLASS FILTERING ===
CLASSES_TO_TRACK = ["person", "teleco"]

# === PIXEL BOUNDARY (user-defined on CV window) ===
# Replace with your clicked pixel coordinates
BOUNDARY_POINTS_MAP = np.array([
    [130, 474],
    [189, 92],
    [585, 82],
    [593, 469],
], dtype=np.int32)

# camera → map transform (for deprojected points)
T_MAP_CAM = np.array([
  [ 0.10099723, -0.36969855,  0.92364633, -3.98448572],
  [-0.99488402, -0.03537113,  0.09462916,  1.35687245],
  [-0.00231385, -0.92847825, -0.37137957,  1.35620029],
  [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
], dtype=float)


def point_in_polygon(pt, poly):
    """Ray-casting algorithm for 2D integer polygon"""
    x, y = pt
    inside = False
    n = len(poly)
    p1x, p1y = poly[0]
    for i in range(1, n + 1):
        p2x, p2y = poly[i % n]
        if (y > min(p1y, p2y)) and (y <= max(p1y, p2y)) and (x <= max(p1x, p2x)):
            if p1y != p2y:
                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or x <= xinters:
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

class HumanPublisher:
    def __init__(self):
        # MQTT
        self.mqtt = MQTT()
        threading.Thread(target=self.mqtt.connect, daemon=True).start()
        time.sleep(1.0)

        # YOLO
        self.model = YOLO(MODEL_PATH)
        self.model.fuse()
        self.model = self.model.to('cuda').half()
        self.class_indices = [i for i,name in self.model.names.items() if name in CLASSES_TO_TRACK]
        print(f"Tracking: {[self.model.names[i] for i in self.class_indices]}")

        # RealSense
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, 30)
        profile = self.pipeline.start(cfg)
        self.depth_intr = profile.get_stream(rs.stream.depth) \
                              .as_video_stream_profile() \
                              .get_intrinsics()
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)

        self.running = True
        self.lock = threading.Lock()
        self.latest_dets = []  # list of (Xm, Ym, Zm, cls_name)

        threading.Thread(target=self.inference_loop, daemon=True).start()
        threading.Thread(target=self.publish_loop,   daemon=True).start()

    def inference_loop(self):
        sx, sy = FRAME_W/DET_W, FRAME_H/DET_H
        cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
        while self.running:
            frames  = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            cf = aligned.get_color_frame(); df = aligned.get_depth_frame()
            if not cf or not df:
                continue
            img   = np.asanyarray(cf.get_data())
            depth = cv2.medianBlur(np.asanyarray(df.get_data()), 5)
            small = cv2.resize(img, (DET_W, DET_H))

            res = self.model.track(
                small, conf=CONF_THRESH, iou=IOU_THRESH,
                tracker=TRACKER_CFG, persist=True,
                classes=self.class_indices
            )[0]

            vis  = img.copy()
            # draw pixel boundary
            cv2.polylines(vis, [BOUNDARY_POINTS_MAP.reshape(-1,1,2)], True, (0,255,0), 2)

            dets = []
            for box in res.boxes:
                x1,y1,x2,y2 = box.xyxy[0].cpu().numpy()
                x1,x2 = int(x1*sx), int(x2*sx)
                y1,y2 = int(y1*sy), int(y2*sy)
                cx, cy = (x1+x2)//2, (y1+y2)//2

                # gate by pixel boundary
                if not point_in_polygon((cx, cy), BOUNDARY_POINTS_MAP):
                    continue

                # depth and map transform
                patch = depth[max(0,cy-3):min(FRAME_H,cy+4), max(0,cx-3):min(FRAME_W,cx+4)]
                if patch.size == 0: continue
                z_cam = float(np.median(patch)) * self.depth_scale
                if z_cam <= 0: continue

                Xc,Yc,Zc = rs.rs2_deproject_pixel_to_point(self.depth_intr, [cx,cy], z_cam)
                P = T_MAP_CAM @ np.array([Xc,Yc,Zc,1.0], float)
                Xm, Ym, Zm = P[:3]
                dets.append((float(Xm), float(Ym), float(Zm), self.model.names[int(box.cls[0])]))

                # draw detection
                cls_name = self.model.names[int(box.cls[0])]
                color = (255,0,0) if cls_name=='person' else (0,0,255)
                cv2.rectangle(vis, (x1,y1), (x2,y2), color, 2)
                cv2.putText(vis, f"{cls_name} ({Xm:.2f},{Ym:.2f},{Zm:.2f})",
                           (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            with self.lock:
                self.latest_dets = dets

            cv2.imshow("Detections", vis)
            if cv2.waitKey(1) == 27:
                self.stop(); break

    def publish_loop(self):
        while self.running:
            self.send_people_mqtt()
            time.sleep(0.2)

    def send_people_mqtt(self):
        with self.lock:
            dets = list(self.latest_dets)
        teleco=None; people=[]
        for Xm, Ym, Zm, cls in dets:
            if cls=='teleco': teleco={'x':Xm,'y':Ym,'z':Zm}
            elif cls=='person': people.append({'x':Xm,'y':Ym,'z':Zm})
        msg = {'timestamp':time.time(), 'frame_id':'map', 'teleco':teleco, 'people':people}
        try:
            self.mqtt.publish_human_results(json.dumps(msg))
        except Exception as e:
            print("[mqtt error]", e)

    def stop(self):
        self.running=False
        self.pipeline.stop()
        cv2.destroyAllWindows()

if __name__=='__main__':
    pub = HumanPublisher()
    try:
        while pub.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pub.stop()
