#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import math

# === CONFIGURATION: edit these only ===
MODEL_PATH     = '/home/commu/Desktop/human_detector_ws/models/best_yolo11s.pt'

# YOLO tracking
#   conf      = 0.6    # Detection confidence threshold: 
#                      #   ↑ higher → fewer false positives, but may miss small/occluded objects
#                      #   ↓ lower  → more detections, but more noise
#
#   iou       = 0.7    # NMS IoU threshold:
#                      #   ↓ lower → stricter merging, less box overlap
#                      #   ↑ higher→ allow closer boxes, may keep duplicates
CONF_THRESH    = 0.4      # confidence threshold
IOU_THRESH     = 0.6      # NMS IoU threshold
FRAME_WIDTH    = 640      # RealSense color/depth width
FRAME_HEIGHT   = 480      # RealSense color/depth height
USE_TRACKING   = True     # False = detect only, True = track()
TRACKER_CONFIG = 'bytetrack.yaml'


# === YOUR CAMERA→MAP TRANSFORM (replace with your computed matrix) ===
T_MAP_CAM = np.array([
    [-0.99031201,  0.05496663,  0.12751783, -1.90818202],
    [-0.12520336,  0.04368657, -0.99116881,  4.72423410],
    [-0.06005202, -0.99753203, -0.03638133,  0.28598173],
    [0.0,          0.0,          0.0,          1.0       ]
], dtype=float)

# ===================================================================

def main():
    model = YOLO(MODEL_PATH)

    class_colors = {
        model.names[0]: (255,   0, 255),  
        model.names[1]: (  0,   0, 255),
    }
    z_bg_color = (0, 0, 0)

    # RealSense setup
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 30)
    profile = pipeline.start(cfg)

    depth_sensor  = profile.get_device().first_depth_sensor()
    depth_scale   = depth_sensor.get_depth_scale()
    align         = rs.align(rs.stream.color)
    depth_intr    = profile.get_stream(rs.stream.depth) \
                         .as_video_stream_profile() \
                         .get_intrinsics()

    ema = {}   # smoothing state
    alpha = 0.7
    patch_size = 5
    half       = patch_size // 2

    try:
        while True:
            frames  = pipeline.wait_for_frames()
            aligned = align.process(frames)
            c_frame = aligned.get_color_frame()
            d_frame = aligned.get_depth_frame()
            if not c_frame or not d_frame:
                continue

            img         = np.asanyarray(c_frame.get_data())
            depth_image = np.asanyarray(d_frame.get_data())

            # YOLO detect or track
            if USE_TRACKING:
                results = model.track(
                    img, conf=CONF_THRESH, iou=IOU_THRESH,
                    tracker=TRACKER_CONFIG, persist=True
                )[0]
            else:
                results = model(img, conf=CONF_THRESH, iou=IOU_THRESH)[0]

            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1+x2)//2, (y1+y2)//2

                # median depth patch
                y0, x0 = cy, cx
                y_min, y_max = max(0,y0-half), min(FRAME_HEIGHT, y0+half+1)
                x_min, x_max = max(0,x0-half), min(FRAME_WIDTH,  x0+half+1)
                patch = depth_image[y_min:y_max, x_min:x_max]
                if patch.size == 0: continue
                z_raw = np.median(patch)
                z = float(z_raw) * depth_scale
                if z <= 0: continue

                # back-project to camera frame
                X, Y, Z = rs.rs2_deproject_pixel_to_point(depth_intr, [cx, cy], z)

                # EMA smoothing
                if USE_TRACKING and box.id is not None:
                    tid = int(box.id[0])
                    if tid in ema:
                        prevX, prevY, prevZ = ema[tid]
                        X = alpha*X + (1-alpha)*prevX
                        Y = alpha*Y + (1-alpha)*prevY
                        Z = alpha*Z + (1-alpha)*prevZ
                    ema[tid] = (X, Y, Z)

                # — APPLY CAMERA→MAP TRANSFORM —
                P_cam = np.array([X, Y, Z, 1.0], dtype=float)
                P_map = T_MAP_CAM @ P_cam
                X_map, Y_map, Z_map = P_map[:3]

                # class & draw
                cls_idx  = int(box.cls[0])
                cls_name = model.names[cls_idx]
                color    = class_colors.get(cls_name, (255,255,255))
                cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)

                # labels & overlay
                font, fs, thk = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                txts = [
                    f"{cls_name}",
                    f"Cam X:{X:.2f} Y:{Y:.2f} Z:{Z:.2f}m",
                    f"Map X:{X_map:.2f} Y:{Y_map:.2f} Z:{Z_map:.2f}m"
                ]
                ty = y1 + 20
                for t in txts:
                    cv2.putText(img, t, (x1+5, ty), font, 0.5, (255,255,255), 1, cv2.LINE_AA)
                    ty += 18

            cv2.imshow('YOLO + Depth + Map-Coords', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
