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
POSE_MODEL_PATH = '/home/commu/Desktop/human_detector_ws/models/yolov11n-pose.pt'  # Pose model for person detection and height
FRAME_W, FRAME_H = 640, 480
USE_TRACKING  = True
TRACKER_CFG   = 'bytetrack.yaml'

# YOLO tracking
CONF_THRESH   = 0.4
IOU_THRESH    = 0.8
DET_W, DET_H  = 320, 240
ALPHA_MAP     = 0.85  # EMA smoothing factor for map coordinates

# Only track persons
CLASSES_TO_TRACK = ["person"]

# === BOUNDARY CONFIGURATION ===
# 4‐point polygon in pixel coordinates
BOUNDARY_POINTS_MAP = np.array([
    [176, 392],
    [314, 57],
    [598, 297],
    [479, 479],
], dtype=np.int32)

# camera → map transform (still needed for height estimates)
T_MAP_CAM = np.array([
    [ 0.10099723, -0.36969855,  0.92364633, -3.98448572],
    [-0.99488402, -0.03537113,  0.09462916,  1.35687245],
    [-0.00231385, -0.92847825, -0.37137957,  1.35620029],
    [ 0.        ,  0.        ,  0.        ,  1.        ],
], dtype=float)


def point_in_polygon(point, polygon):
    """Check if a point is inside a polygon using ray casting algorithm"""
    x, y = point
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, len(polygon) + 1):
        p2x, p2y = polygon[i % len(polygon)]
        if (y > min(p1y, p2y)) and (y <= max(p1y, p2y)) and (x <= max(p1x, p2x)):
            if p1y != p2y:
                xinters = (y - p1y)*(p2x - p1x)/(p2y - p1y) + p1x
            if (p1x == p2x) or (x <= xinters):
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def map_to_pixel(map_point, transform_matrix, intrinsics):
    """Convert map coordinates to pixel coordinates for visualization"""
    try:
        T_CAM_MAP = np.linalg.inv(transform_matrix)
        hom = np.array([map_point[0], map_point[1], 0.0, 1.0])
        cam = T_CAM_MAP @ hom
        if cam[2] > 0:
            u, v = rs.rs2_project_point_to_pixel(intrinsics, cam[:3])
            return int(u), int(v)
    except:
        pass
    return None


class PoseHeightDetector:
    def __init__(self):
        self.NOSE_IDX       = 0
        self.LEFT_ANKLE_IDX = 15
        self.RIGHT_ANKLE_IDX= 16

    def calculate_nose_height_3d(self, keypoints, depth_image, depth_intrinsics, depth_scale):
        nose = keypoints[self.NOSE_IDX]
        if len(nose) < 3 or nose[2] < 0.5:
            return None

        nx, ny = int(nose[0]), int(nose[1])
        nd = self._get_depth_at_point(depth_image, nx, ny) * depth_scale
        if nd <= 0:
            return None
        nose_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [nx, ny], nd)

        ground_pts = []
        for idx in (self.LEFT_ANKLE_IDX, self.RIGHT_ANKLE_IDX):
            kp = keypoints[idx]
            if len(kp) >= 3 and kp[2] > 0.5:
                ax, ay = int(kp[0]), int(kp[1])
                ad = self._get_depth_at_point(depth_image, ax, ay) * depth_scale
                if ad > 0:
                    ground_pts.append(rs.rs2_deproject_pixel_to_point(depth_intrinsics, [ax, ay], ad))

        if not ground_pts:
            return None
        avg_ground_y = sum(p[1] for p in ground_pts) / len(ground_pts)
        height = avg_ground_y - nose_3d[1]
        return height if height > 0 else None

    def _get_depth_at_point(self, depth_image, x, y, patch_size=5):
        half = patch_size // 2
        h, w = depth_image.shape
        ymn, ymx = max(0, y-half), min(h, y+half+1)
        xmn, xmx = max(0, x-half), min(w, x+half+1)
        patch = depth_image[ymn:ymx, xmn:xmx]
        return np.median(patch) if patch.size else 0


def draw_text_block(img, text_lines, position, font=cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale=0.45, text_color=(255,255,255),
                   bg_color=(0,0,0,180), thickness=1,
                   padding=6, line_spacing=16):
    x, y = position
    if not text_lines:
        return
    # measure
    max_w = total_h = 0
    heights = []
    for t in text_lines:
        (tw, th), _ = cv2.getTextSize(t, font, font_scale, thickness)
        max_w = max(max_w, tw)
        heights.append(th)
        total_h += line_spacing
    total_h -= (line_spacing - max(heights))

    x1, y1 = x-padding, y-padding
    x2, y2 = x+max_w+padding, y+total_h+padding
    overlay = img.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), bg_color[:3], -1)
    alpha = bg_color[3]/255.0
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

    cy = y + heights[0]
    for t in text_lines:
        cv2.putText(img, t, (x, cy), font, font_scale, text_color, thickness)
        cy += line_spacing


class HumanPublisher:
    def __init__(self):
        # MQTT client
        self.mqtt = MQTT()
        threading.Thread(target=self.mqtt.connect, daemon=True).start()
        time.sleep(1.0)
        # fallback topics if needed
        if not hasattr(self.mqtt, 'human_results_topic'):
            self.mqtt.human_results_topic = 'human/results'

        # load model
        self.model = YOLO(POSE_MODEL_PATH)
        self.model.fuse()
        self.model = self.model.to('cuda').half()
        self.pose_detector = PoseHeightDetector()
        self.class_indices = [0]

        # RealSense
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, 30)
        profile = self.pipeline.start(cfg)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.intr = profile.get_stream(rs.stream.depth)\
                            .as_video_stream_profile()\
                            .get_intrinsics()

        # pixel‐space boundary
        self.boundary_points = BOUNDARY_POINTS_MAP
        print(f"Boundary points (pixel coords):\n{self.boundary_points}")

        # state
        self.running = True
        self.lock = threading.Lock()
        self.latest_dets = []
        self.map_ema = {}

        # threads
        threading.Thread(target=self.inference_loop, daemon=True).start()
        threading.Thread(target=self.publish_loop, daemon=True).start()

    def draw_boundary(self, img):
        """Draw the 4‐point pixel‐boundary directly."""
        pts = self.boundary_points.reshape((-1, 1, 2))
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.1, img, 0.9, 0, img)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)

    def calculate_height_from_keypoints(self, kps, depth):
        return self.pose_detector.calculate_nose_height_3d(
            kps, depth, self.intr, self.depth_scale
        )

    def inference_loop(self):
        cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
        while self.running:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df:
                continue

            img = np.asanyarray(cf.get_data())
            depth = cv2.medianBlur(np.asanyarray(df.get_data()), 5)
            vis = img.copy()

            # draw the pixel‐boundary
            self.draw_boundary(vis)

            res = self.model.track(
                img, conf=CONF_THRESH, iou=IOU_THRESH,
                tracker=TRACKER_CFG, persist=True,
                classes=self.class_indices
            )[0]

            dets = []
            if res.keypoints is not None:
                kps_data = res.keypoints.data.cpu().numpy()
                boxes    = res.boxes.data.cpu().numpy() if res.boxes is not None else []

                for i, kps in enumerate(kps_data):
                    if i >= len(boxes):
                        continue
                    box = boxes[i]
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    tid = int(box[5]) if USE_TRACKING and box.shape[0] > 5 else None

                    # center + depth
                    cx, cy = (x1+x2)//2, (y1+y2)//2
                    patch = depth[max(0,cy-3):min(FRAME_H,cy+4),
                                  max(0,cx-3):min(FRAME_W,cx+4)]
                    if patch.size == 0:
                        continue
                    z = float(np.median(patch)) * self.depth_scale
                    if z <= 0:
                        continue

                    # pixel‐boundary check
                    if not point_in_polygon((cx, cy), self.boundary_points):
                        continue

                    # map‐space projection for height (unchanged)
                    Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(self.intr, [cx, cy], z)
                    P = T_MAP_CAM @ np.array([Xc, Yc, Zc, 1.0], float)
                    Xm_, Ym_, _ = P[:3]

                    height = self.calculate_height_from_keypoints(kps, depth)

                    if tid is not None:
                        prev = self.map_ema.get(tid, np.array([Xm_, Ym_], float))
                        filt = ALPHA_MAP*prev + (1-ALPHA_MAP)*np.array([Xm_, Ym_], float)
                        self.map_ema[tid] = filt
                        Xm, Ym = filt
                    else:
                        Xm, Ym = Xm_, Ym_

                    dets.append((float(Xm), float(Ym), height))

                    # draw box & keypoints
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (147,20,255), 2)
                    for j, (kx, ky, kc) in enumerate(kps):
                        if kc > 0.5:
                            color = (0,0,255) if j == self.pose_detector.NOSE_IDX else (0,255,0)
                            cv2.circle(vis, (int(kx),int(ky)), 3, color, -1)

                    text = [
                        f"Dist: {z:.2f}m",
                        f"Height: {height:.2f}m" if height is not None else "Height: N/A"
                    ]
                    draw_text_block(vis, text, (x1+8, y1+8))

            with self.lock:
                self.latest_dets = dets

            cv2.imshow("Detections", vis)
            if cv2.waitKey(1) == 27:
                self.stop()
                break

    def publish_loop(self):
        while self.running:
            self.send_people_mqtt()
            time.sleep(0.2)

    def send_people_mqtt(self):
        with self.lock:
            dets = list(self.latest_dets)
        people = []
        for x, y, h in dets:
            pd = {"x": x, "y": y, "z": 0.0}
            if h is not None:
                pd["height"] = h
            people.append(pd)
        msg = {"timestamp": time.time(), "frame_id": "map", "people": people}
        try:
            self.mqtt.publish_human_results(json.dumps(msg))
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
