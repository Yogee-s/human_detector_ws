#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import random

# === CONFIGURATION ===
MODEL_PATH     = '/home/commu/Desktop/human_detector_ws/models/best_yolo11s.pt'
CONF_THRESH    = 0.4
IOU_THRESH     = 0.6
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480
USE_TRACKING   = True
TRACKER_CONFIG = 'bytetrack.yaml'

# camera → map transform
T_MAP_CAM = np.array([
    [-0.99031201,  0.05496663,  0.12751783, -1.90818202],
    [-0.12520336,  0.04368657, -0.99116881,  4.72423410],
    [-0.06005202, -0.99753203, -0.03638133,  0.28598173],
    [0.0,          0.0,          0.0,          1.0       ]
], dtype=float)
# =====================

def draw_box_with_xy(img, x1, y1, x2, y2, cls_label, xy, color):
    """
    Draws:
      - a bounding box in `color`
      - a white background behind the text
      - black text: "cls_label X:.. Y:..m"
    """
    Xm, Ym = xy
    # draw box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # prepare label
    text = f"{cls_label} X:{Xm:.2f} Y:{Ym:.2f}m"
    font, scale, thk = cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
    (w, h), _ = cv2.getTextSize(text, font, scale, thk)

    # position: above the box, or below if too high
    tx = x1
    ty = y1 - 10
    if ty - h < 0:
        ty = y1 + h + 10

    # draw solid white background
    cv2.rectangle(
        img,
        (tx - 3, ty - h - 3),
        (tx + w + 3, ty + 3),
        (255, 255, 255),
        cv2.FILLED
    )

    # draw black text
    cv2.putText(img, text, (tx, ty), font, scale, (0, 0, 0), thk, cv2.LINE_AA)


def main():
    model = YOLO(MODEL_PATH)

    # Build color map: teleco=blue, person=red, others=random
    class_colors = {}
    for idx, name in model.names.items():
        lname = name.lower()
        if lname == 'teleco':
            class_colors[name] = (255, 0, 0)    # blue
        elif lname == 'person':
            class_colors[name] = (0, 0, 255)    # red
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

    map_ema   = {}
    alpha_map = 0.6
    patch_half = 3  # for 7×7 depth patch

    try:
        while True:
            frames  = pipeline.wait_for_frames()
            aligned = align.process(frames)
            cf      = aligned.get_color_frame()
            df      = aligned.get_depth_frame()
            if not cf or not df:
                continue

            img   = np.asanyarray(cf.get_data())
            depth = np.asanyarray(df.get_data())
            depth = cv2.medianBlur(depth, 5)

            # detection / tracking
            if USE_TRACKING:
                results = model.track(
                    img, conf=CONF_THRESH, iou=IOU_THRESH,
                    tracker=TRACKER_CONFIG, persist=True
                )[0]
            else:
                results = model(img, conf=CONF_THRESH, iou=IOU_THRESH)[0]

            for box in results.boxes:
                cls_idx  = int(box.cls[0])
                cls_name = model.names[cls_idx]
                color    = class_colors[cls_name]

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                tid = int(box.id[0]) if (USE_TRACKING and box.id is not None) else None

                # bottom-center pixel for floor
                cx = (x1 + x2) // 2
                cy = y2

                patch = depth[
                    max(0, cy - patch_half): cy + patch_half + 1,
                    max(0, cx - patch_half): cx + patch_half + 1
                ]
                if patch.size == 0:
                    continue
                z_raw = float(np.median(patch)) * depth_scale
                if z_raw <= 0:
                    continue

                # back-project to camera frame
                Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], z_raw)

                # camera→map
                P_cam = np.array([Xc, Yc, Zc, 1.0], dtype=float)
                P_map = T_MAP_CAM @ P_cam
                Xm_, Ym_, _ = P_map[:3]
                # floor
                Zm_ = 0.0

                # EMA smoothing on (Xm, Ym)
                if tid is not None:
                    if tid not in map_ema:
                        map_ema[tid] = np.array([Xm_, Ym_])
                    else:
                        map_ema[tid] = (
                            alpha_map * map_ema[tid] +
                            (1 - alpha_map) * np.array([Xm_, Ym_])
                        )
                    Xm, Ym = map_ema[tid]
                else:
                    Xm, Ym = Xm_, Ym_

                draw_box_with_xy(img, x1, y1, x2, y2, cls_name, (Xm, Ym), color)

            cv2.imshow('YOLO + All Classes X/Y in Map', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
