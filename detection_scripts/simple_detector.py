#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import math

# === CONFIGURATION: edit these only ===
MODEL_PATH     = 'models/best_yolo11s.pt'
CONF_THRESH    = 0.6      # confidence threshold
IOU_THRESH     = 0.7      # NMS IoU threshold
FRAME_WIDTH    = 640      # RealSense color/depth width
FRAME_HEIGHT   = 480      # RealSense color/depth height
USE_TRACKING   = True     # False = detect only, True = track()
TRACKER_CONFIG = 'bytetrack.yaml'
# =====================================

def main():
    model = YOLO(MODEL_PATH)

    # fixed colors for classes 0 & 1
    class_colors = {
        model.names[0]: (255,   0, 255),  # purple
        model.names[1]: (  0,   0, 255),  # red
    }
    z_bg_color = (0, 0, 0)  # black for Z label

    # RealSense setup
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 30)
    profile = pipeline.start(cfg)

    # get depth scale
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()

    align = rs.align(rs.stream.color)
    depth_intrinsics = profile.get_stream(rs.stream.depth) \
                              .as_video_stream_profile() \
                              .get_intrinsics()

    # EMA smoothing state per track id
    ema = {}
    alpha = 0.7

    # patch size for depth averaging
    patch_size = 5
    half = patch_size // 2

    try:
        while True:
            # get frames
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            c_frame = aligned.get_color_frame()
            d_frame = aligned.get_depth_frame()
            if not c_frame or not d_frame:
                continue

            img = np.asanyarray(c_frame.get_data())
            depth_image = np.asanyarray(d_frame.get_data())

            # detect or track
            if USE_TRACKING:
                results = model.track(
                    img, conf=CONF_THRESH, iou=IOU_THRESH,
                    tracker=TRACKER_CONFIG, persist=True
                )[0]
            else:
                results = model(img, conf=CONF_THRESH, iou=IOU_THRESH)[0]

            for box in results.boxes:
                # bounding box coords
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2)//2, (y1 + y2)//2

                # patch median depth
                y0, x0 = cy, cx
                y_min, y_max = max(0, y0-half), min(FRAME_HEIGHT, y0+half+1)
                x_min, x_max = max(0, x0-half), min(FRAME_WIDTH,  x0+half+1)
                patch = depth_image[y_min:y_max, x_min:x_max]
                if patch.size == 0:
                    continue
                median_raw = np.median(patch)
                z = float(median_raw) * depth_scale
                if z <= 0:
                    continue

                # deproject to 3D
                X, Y, Z = rs.rs2_deproject_pixel_to_point(
                    depth_intrinsics, [cx, cy], z
                )

                # class info & color
                cls_idx  = int(box.cls[0])
                cls_name = model.names[cls_idx]
                color    = class_colors.get(cls_name, (255,255,255))

                # EMA smoothing on world coords
                if USE_TRACKING and box.id is not None:
                    tid = int(box.id[0])
                    prev = ema.get(tid, None)
                    if prev:
                        X = alpha * X + (1-alpha) * prev[0]
                        Y = alpha * Y + (1-alpha) * prev[1]
                        Z = alpha * Z + (1-alpha) * prev[2]
                    ema[tid] = (X, Y, Z)

                # draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # — SHOW CLASS NAME INSIDE BOX —
                font, fs, thk = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
                (cls_w, cls_h), _ = cv2.getTextSize(cls_name, font, fs, thk)
                lx1, ly1 = x1 + 2, y1 + 2
                lx2, ly2 = lx1 + cls_w + 4, ly1 + cls_h + 4
                cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, cv2.FILLED)
                cv2.putText(img, cls_name, (lx1 + 2, ly2 - 2),
                            font, fs, (0,0,0), thk, cv2.LINE_AA)

                # — XYZ OVERLAY INSIDE BOX, BELOW CLASS —
                labels = [f"X:{X:.2f}", f"Y:{Y:.2f}", f"Z:{Z:.2f}m"]
                font2, fs2, thk2 = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                spacing = 6
                tx, ty = lx1 + 2, ly2 + cls_h + 4
                for lbl in labels:
                    (w, h), _ = cv2.getTextSize(lbl, font2, fs2, thk2)
                    bgc = z_bg_color if lbl.startswith("Z:") else color
                    cv2.rectangle(
                        img,
                        (tx - 2, ty - h - 2),
                        (tx + w + 2, ty + 2),
                        bgc, cv2.FILLED
                    )
                    cv2.putText(
                        img, lbl,
                        (tx, ty),
                        font2, fs2, (255,255,255), thk2
                    )
                    tx += w + spacing

            cv2.imshow('YOLO + Depth + Smoothed XYZ', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
