#!/usr/bin/env python3
import threading
import time
import json
import math

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
# TRACKER_CFG = 'botsort.yaml'  

# YOLO tracking
CONF_THRESH   = 0.4
IOU_THRESH    = 0.8
DET_W, DET_H  = 320, 240
ALPHA_MAP     = 0.75  # Reduced EMA smoothing factor for map coordinates

# Smoothing parameters - reduced for better multi-person handling
ALPHA_DETECTION = 0.5  # Reduced EMA smoothing factor for detection stability

# Only track persons
CLASSES_TO_TRACK = ["person"]

# === BOUNDARY CONFIGURATION ===
# 4‐point polygon in pixel coordinates
BOUNDARY_POINTS_MAP = np.array([
    [63, 429],
    [202, 197],
    [385, 201],
    [419, 449],
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

        # Point selection
        self.selected_point = None  # Will store (x, y) in map coordinates
        self.selected_pixel = None  # Will store (x, y) in pixel coordinates for visualization
        
        # Detection smoothing - improved for multi-person scenarios
        self.detection_smoothing = {}  # track_id -> smoothed values
        self.detection_history = {}   # track_id -> history of detections
        self.max_history_length = 5  # Keep last 5 detections for validation

        # state
        self.running = True
        self.lock = threading.Lock()
        self.latest_dets = []
        self.map_ema = {}
        self.nearest_person = None
        self.nearest_person_index = None  # NEW: Store the index of nearest person
        self.current_depth = None

        # threads
        threading.Thread(target=self.inference_loop, daemon=True).start()
        threading.Thread(target=self.publish_loop, daemon=True).start()

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks to select a point"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Convert pixel coordinates to map coordinates
            try:
                with self.lock:
                    if self.current_depth is not None:
                        depth_value = self.current_depth[y, x] * self.depth_scale
                        if depth_value > 0:
                            # Convert to 3D camera coordinates
                            Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(self.intr, [x, y], depth_value)
                            # Transform to map coordinates
                            P = T_MAP_CAM @ np.array([Xc, Yc, Zc, 1.0], float)
                            map_x, map_y = P[0], P[1]
                            
                            self.selected_point = (map_x, map_y)
                            self.selected_pixel = (x, y)
                            print(f"Selected point: Pixel({x}, {y}) -> Map({map_x:.2f}, {map_y:.2f})")
                        else:
                            print("No valid depth at clicked point")
            except Exception as e:
                print(f"Error converting clicked point: {e}")

    def validate_detection(self, tid, new_box):
        """Validate detection against history to prevent confusion between people"""
        if tid not in self.detection_history:
            self.detection_history[tid] = []
        
        history = self.detection_history[tid]
        
        # If we have history, check if new detection is reasonable
        if history:
            last_box = history[-1]
            # Calculate movement distance
            cx1, cy1 = (last_box[0] + last_box[2]) // 2, (last_box[1] + last_box[3]) // 2
            cx2, cy2 = (new_box[0] + new_box[2]) // 2, (new_box[1] + new_box[3]) // 2
            movement = math.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
            
            # If movement is too large, it might be a tracking error
            if movement > 100:  # pixels - adjust threshold as needed
                print(f"Warning: Large movement detected for ID {tid}: {movement:.1f} pixels")
                # You could choose to reject this detection or reduce smoothing
                return False
        
        # Add to history
        history.append(new_box)
        if len(history) > self.max_history_length:
            history.pop(0)
        
        return True

    def find_nearest_person(self, detections):
        """Find the person nearest to the selected point in map coordinates"""
        if not self.selected_point or not detections:
            return None, None
        
        min_distance = float('inf')
        nearest = None
        nearest_index = None
        
        for i, det in enumerate(detections):
            # Calculate distance in map coordinates
            dx = det['x'] - self.selected_point[0]
            dy = det['y'] - self.selected_point[1]
            distance = math.sqrt(dx*dx + dy*dy)
            
            if distance < min_distance:
                min_distance = distance
                nearest = det
                nearest_index = i
        
        return nearest, nearest_index

    def draw_boundary(self, img):
        """Draw the 4‐point pixel‐boundary directly."""
        pts = self.boundary_points.reshape((-1, 1, 2))
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
        cv2.addWeighted(overlay, 0.1, img, 0.9, 0, img)
        cv2.polylines(img, [pts], True, (0, 255, 0), 2)

    def draw_selected_point(self, img):
        """Draw the selected point"""
        if self.selected_pixel:
            # Draw selected point
            cv2.circle(img, self.selected_pixel, 8, (0, 0, 255), -1)
            cv2.circle(img, self.selected_pixel, 12, (0, 0, 255), 2)
            
            # Draw text
            cv2.putText(img, "Selected", 
                       (self.selected_pixel[0] + 15, self.selected_pixel[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    def calculate_height_from_keypoints(self, kps, depth):
        return self.pose_detector.calculate_nose_height_3d(
            kps, depth, self.intr, self.depth_scale
        )

    def inference_loop(self):
        cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Detections", self.mouse_callback)
        
        while self.running:
            frames = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df:
                continue

            img = np.asanyarray(cf.get_data())
            depth = cv2.medianBlur(np.asanyarray(df.get_data()), 5)
            
            # Store current depth for mouse callback
            with self.lock:
                self.current_depth = depth.copy()
            
            vis = img.copy()

            # draw the pixel‐boundary
            self.draw_boundary(vis)
            
            # draw selected point
            self.draw_selected_point(vis)

            res = self.model.track(
                img, conf=CONF_THRESH, iou=IOU_THRESH,
                tracker=TRACKER_CFG, persist=True,
                classes=self.class_indices
            )[0]

            dets = []
            detection_objects = []
            
            if res.keypoints is not None:
                kps_data = res.keypoints.data.cpu().numpy()
                boxes    = res.boxes.data.cpu().numpy() if res.boxes is not None else []

                for i, kps in enumerate(kps_data):
                    if i >= len(boxes):
                        continue
                    box = boxes[i]
                    x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                    confidence = float(box[4])
                    tid = int(box[5]) if USE_TRACKING and box.shape[0] > 5 else None

                    # Use bottom middle of bounding box for boundary check
                    cx = (x1 + x2) // 2  # horizontal center
                    # cy = y2  # bottom of bounding box (even if partial human)
                    cy = y2 - 20  # Adjust to avoid edge case at bottom
                    
                    # Validate detection against history
                    if tid is not None and not self.validate_detection(tid, [x1, y1, x2, y2]):
                        continue

                    # Get depth at bottom center
                    patch = depth[max(0,cy-3):min(FRAME_H,cy+4),
                                  max(0,cx-3):min(FRAME_W,cx+4)]
                    if patch.size == 0:
                        continue
                    z = float(np.median(patch)) * self.depth_scale
                    if z <= 0:
                        continue

                    # pixel‐boundary check using bottom middle of bounding box
                    if not point_in_polygon((cx, cy), self.boundary_points):
                        continue

                    # map‐space projection using bottom middle coordinates
                    Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(self.intr, [cx, cy], z)
                    P = T_MAP_CAM @ np.array([Xc, Yc, Zc, 1.0], float)
                    Xm_, Ym_, _ = P[:3]

                    height = self.calculate_height_from_keypoints(kps, depth)

                    # Apply smoothing with improved multi-person handling
                    if tid is not None:
                        prev = self.map_ema.get(tid, np.array([Xm_, Ym_], float))
                        # Use current position if no previous data
                        if tid not in self.map_ema:
                            filt = np.array([Xm_, Ym_], float)
                        else:
                            filt = ALPHA_MAP*prev + (1-ALPHA_MAP)*np.array([Xm_, Ym_], float)
                        self.map_ema[tid] = filt
                        Xm, Ym = filt
                        
                        # Improved bounding box smoothing
                        if tid in self.detection_smoothing:
                            smooth_box = self.detection_smoothing[tid]
                            # Only smooth if the detection seems valid
                            if self.validate_detection(tid, [x1, y1, x2, y2]):
                                x1 = int(ALPHA_DETECTION * smooth_box[0] + (1-ALPHA_DETECTION) * x1)
                                y1 = int(ALPHA_DETECTION * smooth_box[1] + (1-ALPHA_DETECTION) * y1)
                                x2 = int(ALPHA_DETECTION * smooth_box[2] + (1-ALPHA_DETECTION) * x2)
                                y2 = int(ALPHA_DETECTION * smooth_box[3] + (1-ALPHA_DETECTION) * y2)
                        
                        self.detection_smoothing[tid] = [x1, y1, x2, y2]
                    else:
                        Xm, Ym = Xm_, Ym_

                    dets.append((float(Xm), float(Ym), height))
                    
                    # Store detection object for nearest person calculation
                    detection_objects.append({
                        'x': float(Xm),
                        'y': float(Ym),
                        'height': height,
                        'track_id': tid,
                        'confidence': confidence,
                        'box': [x1, y1, x2, y2],
                        'keypoints': kps,
                        'bottom_center': (cx, cy)  # Store center for visualization
                    })

            # Clean up old tracking data
            current_tids = {det['track_id'] for det in detection_objects if det['track_id'] is not None}
            for tid in list(self.map_ema.keys()):
                if tid not in current_tids:
                    del self.map_ema[tid]
                    if tid in self.detection_smoothing:
                        del self.detection_smoothing[tid]
                    if tid in self.detection_history:
                        del self.detection_history[tid]

            # Find nearest person - now returns both person and index
            nearest_person, nearest_index = self.find_nearest_person(detection_objects)
            
            # Draw detections
            for i, det in enumerate(detection_objects):
                x1, y1, x2, y2 = det['box']
                
                # Highlight ONLY the nearest person by comparing index
                if nearest_index is not None and i == nearest_index:
                    color = (0, 255, 255)  # Yellow for nearest
                    thickness = 3
                else:
                    color = (147, 20, 255)  # Default purple
                    thickness = 2
                
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, thickness)
                
                # Draw center point used for boundary check
                cx, cy = det['bottom_center']
                cv2.circle(vis, (cx, cy), 4, (255, 0, 0), -1)  # Blue dot at center
                
                # draw keypoints
                for j, (kx, ky, kc) in enumerate(det['keypoints']):
                    if kc > 0.5:
                        kcolor = (0,0,255) if j == self.pose_detector.NOSE_IDX else (0,255,0)
                        cv2.circle(vis, (int(kx),int(ky)), 3, kcolor, -1)

                # Draw info text
                distance_text = ""
                if nearest_index is not None and i == nearest_index:
                    if self.selected_point:
                        dx = det['x'] - self.selected_point[0]
                        dy = det['y'] - self.selected_point[1]
                        distance = math.sqrt(dx*dx + dy*dy)
                        distance_text = f"Dist: {distance:.2f}m"

                text = [
                    f"ID: {det.get('track_id', 'N/A')}",
                    f"Conf: {det['confidence']:.2f}",
                    f"Height: {det['height']:.2f}m" if det['height'] is not None else "Height: N/A"
                ]
                if distance_text:
                    text.append(distance_text)
                
                draw_text_block(vis, text, (x1+8, y1+8))

            # Draw instructions
            instructions = [
                "Click to select a point",
                "Nearest person highlighted in yellow",
                "Blue dot = center point (boundary check)"
            ]
            draw_text_block(vis, instructions, (10, 10), 
                          text_color=(255, 255, 255), bg_color=(0, 0, 0, 200))

            with self.lock:
                self.latest_dets = dets
                self.nearest_person = nearest_person
                self.nearest_person_index = nearest_index

            cv2.imshow("Detections", vis)
            if cv2.waitKey(1) == 27:
                self.stop()
                break

    def publish_loop(self):
        while self.running:
            self.send_people_mqtt()
            time.sleep(0.2)

    def send_people_mqtt(self):
        """Send only the nearest person to the selected point"""
        with self.lock:
            nearest = self.nearest_person
        
        people = []
        if nearest and self.selected_point:
            pd = {
                "x": nearest['x'], 
                "y": nearest['y'], 
                "z": 0.0,
                "track_id": nearest['track_id'],
                "confidence": nearest['confidence']
            }
            if nearest['height'] is not None:
                pd["height"] = nearest['height']
            
            # Add distance to selected point
            dx = nearest['x'] - self.selected_point[0]
            dy = nearest['y'] - self.selected_point[1]
            pd["distance_to_selected"] = math.sqrt(dx*dx + dy*dy)
            
            people.append(pd)
        
        msg = {
            "timestamp": time.time(), 
            "frame_id": "map", 
            "people": people,
            "selected_point": {
                "x": self.selected_point[0] if self.selected_point else None,
                "y": self.selected_point[1] if self.selected_point else None
            }
        }
        
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
        print("Human detector started. Click on the camera window to select a point.")
        print("Only the nearest person to the selected point will be published via MQTT.")
        print("Green dots show center of bounding boxes (used for boundary check).")
        print("Press ESC to exit.")
        while pub.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pub.stop()


if __name__=='__main__':
    main()