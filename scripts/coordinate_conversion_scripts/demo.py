#!/usr/bin/env python3
import threading
import time

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import random

# === CONFIGURATION ===
MODEL_PATH     = '/home/commu/Desktop/human_detector_ws/models/best_yolo11s.pt'
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
USE_TRACKING   = True
TRACKER_CONFIG = 'bytetrack.yaml'

# Detection thresholds
CONF_THRESH    = 0.5
IOU_THRESH     = 0.4

# Proxy resolution for faster inference
DET_W, DET_H = 320, 240
# EMA smoothing factor (higher → smoother)
ALPHA_MAP = 0.85

# camera → map transform
T_MAP_CAM = np.array([
    [-0.99031201,  0.05496663,  0.12751783, -1.90818202],
    [-0.12520336,  0.04368657, -0.99116881,  4.72423410],
    [-0.06005202, -0.99753203, -0.03638133,  0.28598173],
    [0.0,          0.0,          0.0,          1.0       ]
], dtype=float)
# =====================

def draw_box_with_xy(img, x1, y1, x2, y2, cls_label, xy, color, conf):
    Xm, Ym = xy
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    text1 = f"{cls_label} X:{Xm:.2f} Y:{Ym:.2f}m"
    text2 = f"conf:{conf:.2f}"
    font, scale, thk = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2

    (w1, h1), _ = cv2.getTextSize(text1, font, scale, thk)
    (w2, h2), _ = cv2.getTextSize(text2, font, scale, thk)
    width  = max(w1, w2)
    height = h1 + h2 + 8

    bx = x1
    by = y1 - 10 - height
    if by < 0:
        by = y1 + 10

    cv2.rectangle(
        img,
        (bx - 3, by - 3),
        (bx + width + 3, by + height + 3),
        (255, 255, 255),
        cv2.FILLED
    )

    ty1 = by + h1
    cv2.putText(img, text1, (bx, ty1), font, scale, (0, 0, 0), thk, cv2.LINE_AA)
    ty2 = by + h1 + 8 + h2
    cv2.putText(img, text2, (bx, ty2), font, scale, (0, 0, 0), thk, cv2.LINE_AA)


def main():
    # ——— FIXED MODEL LOADING ———
    model = YOLO(MODEL_PATH)
    model.fuse()                     # in-place fuse; returns None
    model = model.to('cuda').half()  # now move and convert
    # ——————————————

    scale_x = FRAME_WIDTH  / DET_W
    scale_y = FRAME_HEIGHT / DET_H

    # Build color map
    class_colors = {}
    for idx, name in model.names.items():
        lname = name.lower()
        if lname == 'teleco':
            class_colors[name] = (255, 0, 0)
        elif lname == 'person':
            class_colors[name] = (0, 0, 255)
        else:
            class_colors[name] = tuple(random.randint(50, 255) for _ in range(3))

    # RealSense setup
    pipeline = rs.pipeline()
    cfg      = rs.config()
    cfg.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 30)
    profile  = pipeline.start(cfg)

    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    align       = rs.align(rs.stream.color)
    intr        = profile.get_stream(rs.stream.depth) \
                   .as_video_stream_profile() \
                   .get_intrinsics()

    # Shared data and threading
    running = True
    latest_frame = None
    latest_detections = None
    det_lock = threading.Lock()
    map_ema = {}

    def inference_loop():
        nonlocal latest_frame, latest_detections, running
        while running:
            if latest_frame is None:
                time.sleep(0.001)
                continue
            small = cv2.resize(latest_frame, (DET_W, DET_H))
            results = model.track(
                small,
                conf=CONF_THRESH,
                iou=IOU_THRESH,
                tracker=TRACKER_CONFIG,
                persist=True
            )[0]
            boxes = []
            for box in results.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)
                boxes.append({
                    'cls_idx': int(box.cls[0]),
                    'conf': float(box.conf[0]),
                    'xyxy': (x1, y1, x2, y2),
                    'id': int(box.id[0]) if USE_TRACKING and box.id is not None else None
                })
            with det_lock:
                latest_detections = boxes

    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()

    try:
        while True:
            frames  = pipeline.wait_for_frames()
            aligned = align.process(frames)
            cf      = aligned.get_color_frame()
            df      = aligned.get_depth_frame()
            if not cf or not df:
                continue

            img = np.asanyarray(cf.get_data())
            depth = cv2.medianBlur(np.asanyarray(df.get_data()), 5)

            latest_frame = img.copy()
            with det_lock:
                dets = latest_detections

            if dets:
                for det in dets:
                    cls_idx   = det['cls_idx']
                    cls_name  = model.names[cls_idx]
                    color     = class_colors[cls_name]
                    x1, y1, x2, y2 = det['xyxy']
                    conf_score = det['conf']
                    tid = det['id']

                    cx = (x1 + x2) // 2
                    cy = y2
                    patch = depth[
                        max(0, cy - 3): cy + 4,
                        max(0, cx - 3): cx + 4
                    ]
                    if patch.size == 0:
                        continue
                    z_raw = float(np.median(patch)) * depth_scale
                    if z_raw <= 0:
                        continue

                    Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], z_raw)
                    P_map = T_MAP_CAM @ np.array([Xc, Yc, Zc, 1.0], dtype=float)
                    Xm_, Ym_, _ = P_map[:3]

                    if tid is not None:
                        prev = map_ema.get(tid, np.array([Xm_, Ym_]))
                        map_ema[tid] = ALPHA_MAP * prev + (1 - ALPHA_MAP) * np.array([Xm_, Ym_])
                        Xm, Ym = map_ema[tid]
                    else:
                        Xm, Ym = Xm_, Ym_

                    draw_box_with_xy(img, x1, y1, x2, y2, cls_name, (Xm, Ym), color, conf_score)

            cv2.imshow('YOLO Smooth Tracking', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        running = False
        t.join(timeout=1.0)
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
