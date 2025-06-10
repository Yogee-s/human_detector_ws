#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# === CONFIGURATION: edit these only ===
MODEL_PATH     = '/home/commu/Desktop/human_detector_ws/src/human_detector/models/best_yolo11s.pt'
CONF_THRESH    = 0.6      # confidence threshold
IOU_THRESH     = 0.7      # NMS IoU threshold
FRAME_WIDTH    = 640      # RealSense color/depth width
FRAME_HEIGHT   = 480      # RealSense color/depth height
USE_TRACKING   = True     # False = plain detect(), True = track()
TRACKER_CONFIG = 'bytetrack.yaml'
# =====================================

def main():
    # load model
    model = YOLO(MODEL_PATH)

    # generate a distinct color for each class
    colors = {
        name: tuple(int(c) for c in np.random.randint(0, 256, 3))
        for name in model.names.values()
    }

    # RealSense setup
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 30)
    profile = pipeline.start(cfg)
    align   = rs.align(rs.stream.color)
    intr    = profile.get_stream(rs.stream.depth) \
                     .as_video_stream_profile() \
                     .get_intrinsics()

    try:
        while True:
            frames  = pipeline.wait_for_frames()
            aligned = align.process(frames)
            c_frame = aligned.get_color_frame()
            d_frame = aligned.get_depth_frame()
            if not c_frame or not d_frame:
                continue

            img = np.asanyarray(c_frame.get_data())

            # pick detect vs track
            if USE_TRACKING:
                results = model.track(
                    img,
                    conf=CONF_THRESH,
                    iou=IOU_THRESH,
                    tracker=TRACKER_CONFIG,
                    persist=True
                )[0]
            else:
                results = model(
                    img,
                    conf=CONF_THRESH,
                    iou=IOU_THRESH
                )[0]

            # draw every box + class + distance
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cx, cy = (x1 + x2)//2, (y1 + y2)//2

                z = d_frame.get_distance(cx, cy)
                if z <= 0:
                    continue

                cls_idx  = int(box.cls[0])
                cls_name = model.names[cls_idx]
                color    = colors[cls_name]

                # bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

                # label = “Class XX.XXm”
                label = f"{cls_name} {z:.2f}m"
                font, fs, thk = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                (tw, th), _ = cv2.getTextSize(label, font, fs, thk)

                # filled background INSIDE box
                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x1 + tw + 4, y1 + th + 4),
                    color, cv2.FILLED
                )
                # white text
                cv2.putText(
                    img, label,
                    (x1 + 2, y1 + th + 2),
                    font, fs, (255, 255, 255), thk
                )

            cv2.imshow('YOLO + Depth', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
