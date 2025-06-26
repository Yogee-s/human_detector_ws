#!/usr/bin/env python3
import threading
import time
import json

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import torch

from mqtt_client import MQTT

torch.backends.cudnn.benchmark = True

# === CONFIGURATION ===
MODEL_PATH   = '/home/commu/Desktop/human_detector_ws/models/best_yolo11m.pt'
FRAME_W, FRAME_H = 640, 480
USE_TRACKING = True
TRACKER_CFG  = 'bytetrack.yaml'
CONF_THRESH  = 0.5
IOU_THRESH   = 0.4
DET_W, DET_H = 320, 240
ALPHA_MAP    = 0.85

# camera → map transform
T_MAP_CAM = np.array([
    [-0.99031201,  0.05496663,  0.12751783, -1.90818202],
    [-0.12520336,  0.04368657, -0.99116881,  4.72423410],
    [-0.06005202, -0.99753203, -0.03638133,  0.28598173],
    [0.0,          0.0,          0.0,          1.0       ]
], dtype=float)


class HumanPublisher:
    def __init__(self):
        # start MQTT client in its own thread so loop_forever() doesn't block us
        self.mqtt = MQTT()
        threading.Thread(target=self.mqtt.connect, daemon=True).start()
        # give it a moment to connect
        time.sleep(1.0)

        # — YOLO model —
        self.model = YOLO(MODEL_PATH)
        self.model.fuse()
        self.model = self.model.to('cuda').half()

        # — RealSense setup —
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, 30)
        profile = self.pipeline.start(cfg)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.intr = profile.get_stream(rs.stream.depth) \
                       .as_video_stream_profile() \
                       .get_intrinsics()

        # — shared state —
        self.running     = True
        self.lock        = threading.Lock()
        self.latest_dets = []      # list of (x,y)
        self.map_ema     = {}      # for smoothing by ID

        # — start threads —
        threading.Thread(target=self.inference_loop, daemon=True).start()
        threading.Thread(target=self.publish_loop,   daemon=True).start()

    def inference_loop(self):
        sx = FRAME_W / DET_W
        sy = FRAME_H / DET_H

        while self.running:
            try:
                frames  = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)
                cf      = aligned.get_color_frame()
                df      = aligned.get_depth_frame()
                if not cf or not df:
                    continue

                img   = np.asanyarray(cf.get_data())
                depth = cv2.medianBlur(np.asanyarray(df.get_data()), 5)
                small = cv2.resize(img, (DET_W, DET_H))

                res = self.model.track(
                    small,
                    conf=CONF_THRESH,
                    iou=IOU_THRESH,
                    tracker=TRACKER_CFG,
                    persist=True
                )[0]

                dets = []
                for box in res.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, x2 = int(x1*sx), int(x2*sx)
                    y1, y2 = int(y1*sy), int(y2*sy)

                    tid = int(box.id[0]) if USE_TRACKING and box.id is not None else None
                    cx, cy = (x1 + x2)//2, y2

                    # depth patch
                    y0, y1 = max(0, cy-3), min(FRAME_H, cy+4)
                    x0, x1 = max(0, cx-3), min(FRAME_W, cx+4)
                    patch = depth[y0:y1, x0:x1]
                    if patch.size == 0:
                        continue

                    z = float(np.median(patch)) * self.depth_scale
                    if z <= 0:
                        continue

                    Xc,Yc,Zc = rs.rs2_deproject_pixel_to_point(self.intr, [cx,cy], z)
                    P = T_MAP_CAM @ np.array([Xc,Yc,Zc,1.0], float)
                    Xm_, Ym_, _ = P[:3]

                    if tid is not None:
                        prev = self.map_ema.get(tid, np.array([Xm_,Ym_],float))
                        filt = ALPHA_MAP * prev + (1-ALPHA_MAP) * np.array([Xm_,Ym_],float)
                        self.map_ema[tid] = filt
                        Xm, Ym = filt
                    else:
                        Xm, Ym = Xm_, Ym_

                    dets.append((float(Xm), float(Ym)))

                with self.lock:
                    self.latest_dets = dets

            except Exception as e:
                print(f"[inference error] {e}")
                time.sleep(0.1)

    def publish_loop(self):
        while self.running:
            self.send_people_mqtt()
            time.sleep(0.05)   # ~20 Hz

    def send_people_mqtt(self):
        with self.lock:
            dets = list(self.latest_dets)

        msg = {
            "timestamp": time.time(),
            "frame_id":  "map",
            "people":    [{"x": x, "y": y, "z": 0.0} for x, y in dets]
        }

        try:
            self.mqtt.publish_human_results(json.dumps(msg))
            print(f"[mqtt] published {len(dets)} human(s)")
        except Exception as e:
            print(f"[mqtt error] {e}")

    def stop(self):
        print("Shutting down...")
        self.running = False
        try:
            self.pipeline.stop()
            # if helper exposed a disconnect, call it here:
            # self.mqtt.disconnect()
        except:
            pass

def main():
    pub = HumanPublisher()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pub.stop()

if __name__ == '__main__':
    main()
