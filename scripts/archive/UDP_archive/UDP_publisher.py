#!/usr/bin/env python3
import time
import socket
import json

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import torch

torch.backends.cudnn.benchmark = True

# === CONFIGURATION ===
MODEL_PATH    = '/home/commu/Desktop/human_detector_ws/models/best_yolo11m.pt'
FRAME_W, FRAME_H = 640, 480
USE_TRACKING  = True
TRACKER_CFG   = 'bytetrack.yaml'
CONF_THRESH   = 0.5
IOU_THRESH    = 0.4
DET_W, DET_H  = 320, 240
ALPHA_MAP     = 0.85

# UDP config
ROBOT_IP   = "192.168.10.3"
UDP_PORT   = 12345

# camera → map transform
T_MAP_CAM = np.array([
    [-0.99031201,  0.05496663,  0.12751783, -1.90818202],
    [-0.12520336,  0.04368657, -0.99116881,  4.72423410],
    [-0.06005202, -0.99753203, -0.03638133,  0.28598173],
    [0.0,          0.0,          0.0,          1.0       ]
], dtype=float)

def main():
    # --- init model ---
    model = YOLO(MODEL_PATH)
    model.fuse()
    model = model.to('cuda').half()
    names = model.names  # class id → name

    # --- init RealSense ---
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    intr = profile.get_stream(rs.stream.depth) \
              .as_video_stream_profile() \
              .get_intrinsics()

    # --- init UDP socket ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    robot_addr = (ROBOT_IP, UDP_PORT)

    # create display window
    cv2.namedWindow("Humans", cv2.WINDOW_NORMAL)

    last_udp = 0.0
    try:
        while True:
            frames  = pipeline.wait_for_frames()
            aligned = align.process(frames)
            cf      = aligned.get_color_frame()
            df      = aligned.get_depth_frame()
            if not cf or not df:
                continue

            color = np.asanyarray(cf.get_data())
            depth = cv2.medianBlur(np.asanyarray(df.get_data()), 5)

            # down-sample for detection/tracking
            small = cv2.resize(color, (DET_W, DET_H))
            res   = model.track(
                        small,
                        conf=CONF_THRESH,
                        iou=IOU_THRESH,
                        tracker=TRACKER_CFG,
                        persist=True
                    )[0]

            detections = []
            sx, sy = FRAME_W/DET_W, FRAME_H/DET_H

            for box in res.boxes:
                # get bbox in full-res coords
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1, x2 = int(x1*sx), int(x2*sx)
                y1, y2 = int(y1*sy), int(y2*sy)

                # class & track id
                cls_id = int(box.cls[0].cpu().item()) if hasattr(box, 'cls') else -1
                name   = names.get(cls_id, f"id{cls_id}")
                tid    = int(box.id[0]) if USE_TRACKING and box.id is not None else -1

                # pick a small patch at foot center for depth
                cx, cy = (x1+x2)//2, y2
                y0, y1_ = max(0, cy-3), min(FRAME_H, cy+4)
                x0, x1_ = max(0, cx-3), min(FRAME_W, cx+4)
                patch = depth[y0:y1_, x0:x1_]
                if patch.size == 0:
                    continue
                z = float(np.median(patch)) * depth_scale
                if z <= 0:
                    continue

                # deproject → map
                Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], z)
                P = T_MAP_CAM @ np.array([Xc, Yc, Zc, 1.0], float)
                Xm, Ym, Zm = P[:3]

                detections.append({
                    "name": name,
                    "tid": tid,
                    "Xm": Xm, "Ym": Ym, "Zm": Zm,
                    "bbox": (x1, y1, x2, y2)
                })

            # draw detections
            vis = color.copy()
            for det in detections:
                x1, y1, x2, y2 = det["bbox"]
                # choose color by class
                col = (255,128,0) if det["name"]=="teleco" else (0,255,0)
                cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
                label = ( f"{det['name']}#{det['tid']} "
                          f"({det['Xm']:.2f},{det['Ym']:.2f},{det['Zm']:.2f})" )
                cv2.putText(vis, label, (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)

            cv2.imshow("Humans", vis)
            if cv2.waitKey(1) == 27:  # press ESC to quit
                break

            # send UDP at ~20 Hz
            now = time.time()
            if now - last_udp > 0.05:
                msg = {
                    "timestamp": now,
                    "frame_id":  "map",
                    "people": [
                        {"id": det["tid"],
                         "class": det["name"],
                         "x": det["Xm"], "y": det["Ym"], "z": det["Zm"]}
                        for det in detections
                    ]
                }
                sock.sendto(json.dumps(msg).encode('utf-8'), robot_addr)
                last_udp = now

    except KeyboardInterrupt:
        pass
    finally:
        pipeline.stop()
        sock.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
