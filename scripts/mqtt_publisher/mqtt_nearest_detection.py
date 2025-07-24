#!/usr/bin/env python3
import threading
import time
import json
import math
import sys

import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import torch

from helper_scripts.mqtt_client import MQTT

torch.backends.cudnn.benchmark = True

# === CONFIGURATION ===
POSE_MODEL_PATH            = '/home/commu/Desktop/human_detector_ws/models/yolo11m-pose.pt'
# FRAME_W, FRAME_H           = 640, 480
FRAME_W, FRAME_H = 1280, 720
CONF_THRESH                = 0.4
IOU_THRESH                 = 0.8

# 4-point pixel boundary polygon
BOUNDARY_POINTS_MAP = np.array([
    [208, 677],
    [432, 256],
    [733, 260],
    [792, 699],
], dtype=np.int32)

# Camera→map transform for SLAM point projection
T_MAP_CAM = np.array([
  [-0.15253896,  0.52898873, -0.83480703,  2.07710514],
  [ 0.98813506,  0.09694561, -0.11912449,  0.81320340],
  [ 0.01791536, -0.84307323, -0.53750030,  1.40365008],
  [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
], dtype=float)


DEFAULT_SLAM_POINT         = (-0.9, 0.0)
SLAM_POINT_VERTICAL_OFFSET = 0.2  # meters

def point_in_polygon(point, polygon):
    x, y = point
    inside = False
    p1x, p1y = polygon[0]
    for i in range(1, len(polygon)+1):
        p2x, p2y = polygon[i % len(polygon)]
        if (y > min(p1y,p2y)) and (y <= max(p1y,p2y)) and (x <= max(p1x,p2x)):
            if p1y != p2y:
                xinters = (y-p1y)*(p2x-p1x)/(p2y-p1y) + p1x
            if (p1x == p2x) or (x <= xinters):
                inside = not inside
        p1x, p1y = p2x, p2y
    return inside

def map_to_pixel(map_point, transform_matrix, intrinsics):
    try:
        T_CAM_MAP = np.linalg.inv(transform_matrix)
        hom = np.array([map_point[0], map_point[1], -SLAM_POINT_VERTICAL_OFFSET, 1.0])
        cam = T_CAM_MAP @ hom
        if cam[2] > 0:
            u, v = rs.rs2_project_point_to_pixel(intrinsics, cam[:3])
            return int(u), int(v)
    except:
        pass
    return None

def parse_slam_point(args):
    if len(args) >= 3:
        try:
            return float(args[1]), float(args[2])
        except ValueError:
            print(f"Invalid coords '{args[1]}','{args[2]}'; using default {DEFAULT_SLAM_POINT}")
    else:
        print(f"No SLAM point args; using default {DEFAULT_SLAM_POINT}")
    return DEFAULT_SLAM_POINT

class PoseHeightDetector:
    def __init__(self):
        self.NOSE_IDX        = 0
        self.LEFT_ANKLE_IDX  = 15
        self.RIGHT_ANKLE_IDX = 16

    def get_ankle_midpoint(self, keypoints):
        """Get the midpoint between left and right ankles, with fallback options"""
        left_ankle = keypoints[self.LEFT_ANKLE_IDX]
        right_ankle = keypoints[self.RIGHT_ANKLE_IDX]
        
        # Check if both ankles are detected with good confidence
        left_valid = len(left_ankle) >= 3 and left_ankle[2] > 0.5
        right_valid = len(right_ankle) >= 3 and right_ankle[2] > 0.5
        
        if left_valid and right_valid:
            # Both ankles detected - use midpoint
            mid_x = (left_ankle[0] + right_ankle[0]) / 2
            mid_y = (left_ankle[1] + right_ankle[1]) / 2
            return int(mid_x), int(mid_y), "both_ankles"
        elif left_valid:
            # Only left ankle detected
            return int(left_ankle[0]), int(left_ankle[1]), "left_ankle"
        elif right_valid:
            # Only right ankle detected
            return int(right_ankle[0]), int(right_ankle[1]), "right_ankle"
        else:
            # No ankles detected - return None
            return None, None, "no_ankles"

    def get_nose_aligned_bottom_fallback(self, keypoints, bounding_box):
        """Get nose-aligned bottom middle of bounding box as fallback position"""
        nose = keypoints[self.NOSE_IDX]
        x1, y1, x2, y2 = bounding_box
        
        # Check if nose is detected with good confidence
        if len(nose) >= 3 and nose[2] > 0.5:
            # Use nose X coordinate aligned with bottom of bounding box
            cx = int(nose[0])
            cy = y2 - 20  # Bottom of box minus 20 pixels
            return cx, cy, "nose_aligned_bottom"
        else:
            # Final fallback to center of box
            cx = (x1 + x2) // 2
            cy = y2 - 20
            return cx, cy, "box_center"

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

def draw_text_block(img, lines, pos, font=cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale=0.45, text_color=(255,255,255),
                    bg_color=(0,0,0,180), thickness=1,
                    padding=6, line_spacing=16):
    x, y = pos
    if not lines:
        return
    max_w = total_h = 0
    heights = []
    for t in lines:
        (tw, th), _ = cv2.getTextSize(t, font, font_scale, thickness)
        max_w = max(max_w, tw)
        heights.append(th)
        total_h += line_spacing
    total_h -= (line_spacing - heights[-1])

    x1, y1 = x-padding, y-padding
    x2, y2 = x+max_w+padding, y+total_h+padding
    overlay = img.copy()
    cv2.rectangle(overlay, (x1,y1), (x2,y2), bg_color[:3], -1)
    alpha = bg_color[3]/255.0
    cv2.addWeighted(overlay, alpha, img, 1-alpha, 0, img)

    cy = y + heights[0]
    for t in lines:
        cv2.putText(img, t, (x, cy), font, font_scale, text_color, thickness)
        cy += line_spacing

def draw_modern_header(img, slam_point, detection_count, nearest_person, header_height=100):
    """Draw a modern, aesthetic header with key information"""
    width = img.shape[1]
    
    # Create gradient background
    overlay = img.copy()
    for i in range(header_height):
        alpha = 0.95 - (i / header_height) * 0.3  # Gradient fade
        color_intensity = int(25 + (i / header_height) * 15)  # Slight gradient
        cv2.line(overlay, (0, i), (width, i), (color_intensity, color_intensity, color_intensity), 1)
    
    # Blend overlay
    cv2.addWeighted(overlay, 0.9, img, 0.1, 0, img)
    
    # Add subtle accent line at bottom
    cv2.line(img, (0, header_height-2), (width, header_height-2), (70, 170, 255), 3)
    
    # Left side - SLAM info, detection count, and position method
    slam_text = f"SLAM: ({slam_point[0]:.1f}, {slam_point[1]:.1f})"
    cv2.putText(img, slam_text, (20, 30), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)
    
    detect_text = f"Detected: {detection_count}"
    cv2.putText(img, detect_text, (20, 55), cv2.FONT_HERSHEY_DUPLEX, 0.6, (180, 255, 180), 2)
    
    # Position method for nearest person - more prominent
    if nearest_person:
        method_display = {
            "both_ankles": "Both Ankles",
            "left_ankle": "Left Ankle", 
            "right_ankle": "Right Ankle",
            "nose_aligned_bottom": "Nose-Bottom",
            "box_center": "Box Center"
        }
        method_text = f"Method: {method_display.get(nearest_person['positioning_method'], 'Unknown')}"
        cv2.putText(img, method_text, (20, 80), cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 100), 2)
    
    # Right side - Height and Distance display (single line format)
    if nearest_person and nearest_person.get('height') is not None:
        height_m = nearest_person['height']
        height_text = f"Height: {height_m:.2f}m"
        
        # Calculate distance
        dist = math.hypot(nearest_person['x'] - slam_point[0], nearest_person['y'] - slam_point[1])
        dist_text = f"Distance: {dist:.1f}m"
        
        # Calculate sizes for longer boxes with larger text
        height_text_size = cv2.getTextSize(height_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
        height_box_width = height_text_size[0] + 30
        height_box_height = 50
        
        dist_text_size = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
        dist_box_width = dist_text_size[0] + 30
        dist_box_height = 50
        
        # Position boxes side by side
        spacing = 15
        total_width = height_box_width + spacing + dist_box_width
        start_x = width - total_width - 20
        box_y = 25
        
        # Height box
        height_box_x = start_x
        overlay = img.copy()
        cv2.rectangle(overlay, (height_box_x, box_y), (height_box_x + height_box_width, box_y + height_box_height), (0, 150, 255), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cv2.rectangle(img, (height_box_x, box_y), (height_box_x + height_box_width, box_y + height_box_height), (100, 200, 255), 2)
        
        # Height text (single line with black labels)
        label_size = cv2.getTextSize("Height:", cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
        # Draw label in black
        cv2.putText(img, "Height:", (height_box_x + 10, box_y + 32), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3)
        # Draw value part in bright white
        value_x = height_box_x + 10 + label_size[0] + 8
        cv2.putText(img, f"{height_m:.2f}m", (value_x, box_y + 32), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
        
        # Distance box - back to original darker color
        dist_box_x = height_box_x + height_box_width + spacing
        overlay = img.copy()
        cv2.rectangle(overlay, (dist_box_x, box_y), (dist_box_x + dist_box_width, box_y + dist_box_height), (255, 150, 0), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        cv2.rectangle(img, (dist_box_x, box_y), (dist_box_x + dist_box_width, box_y + dist_box_height), (255, 200, 100), 2)
        
        # Distance text (single line with black labels)
        dist_label_size = cv2.getTextSize("Distance:", cv2.FONT_HERSHEY_DUPLEX, 1.2, 3)[0]
        # Draw label in black
        cv2.putText(img, "Distance:", (dist_box_x + 10, box_y + 32), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3)
        # Draw value part in bright white
        dist_value_x = dist_box_x + 10 + dist_label_size[0] + 8
        cv2.putText(img, f"{dist:.1f}m", (dist_value_x, box_y + 32), cv2.FONT_HERSHEY_DUPLEX, 1.2, (255, 255, 255), 3)
        
    else:
        # No height/distance available
        no_data_text = "No Person Data"
        text_size = cv2.getTextSize(no_data_text, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
        text_x = width - text_size[0] - 20
        cv2.putText(img, no_data_text, (text_x, 50), cv2.FONT_HERSHEY_DUPLEX, 0.8, (150, 150, 150), 2)

class HumanPublisher:
    def __init__(self, slam_point):
        self.slam_point = slam_point
        print(f"Target SLAM point: ({slam_point[0]:.2f}, {slam_point[1]:.2f})")

        # MQTT
        self.mqtt = MQTT()
        threading.Thread(target=self.mqtt.connect, daemon=True).start()
        time.sleep(1.0)
        if not hasattr(self.mqtt, 'human_results_topic'):
            self.mqtt.human_results_topic = 'human/results'

        # load YOLO
        self.model = YOLO(POSE_MODEL_PATH)
        self.model.fuse()
        self.model.to('cuda')
        self.model.half()

        self.pose_detector = PoseHeightDetector()

        # RealSense setup
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

        self.boundary_points = BOUNDARY_POINTS_MAP
        self.slam_pixel = map_to_pixel(self.slam_point, T_MAP_CAM, self.intr)
        if self.slam_pixel:
            print(f"SLAM pixel: {self.slam_pixel}")
        else:
            print("SLAM point not visible in view")

        self.running        = True
        self.lock           = threading.Lock()
        self.latest_dets    = []
        self.nearest_person = None
        self.all_detections = []

        threading.Thread(target=self.inference_loop, daemon=True).start()
        threading.Thread(target=self.publish_loop, daemon=True).start()

    def find_nearest_person(self, detections):
        if not detections:
            return None, None
        min_d, best_idx = float('inf'), None
        best_det = None
        for i, det in enumerate(detections):
            d = math.hypot(det['x']-self.slam_point[0], det['y']-self.slam_point[1])
            if d < min_d:
                min_d, best_idx, best_det = d, i, det
        return best_det, best_idx

    def draw_boundary(self, img):
        pts = self.boundary_points.reshape((-1,1,2))
        overlay = img.copy()
        cv2.fillPoly(overlay, [pts], (0,255,0))
        cv2.addWeighted(overlay, 0.1, img, 0.9, 0, img)
        cv2.polylines(img, [pts], True, (0,255,0), 2)

    def draw_slam_point(self, img):
        if not self.slam_pixel:
            return
        x, y = self.slam_pixel

        # draw on overlay for transparency
        overlay = img.copy()
        # small filled dot
        cv2.circle(overlay, (x, y), 3, (255, 255, 0), -1)
        # thin outline circle
        cv2.circle(overlay, (x, y), 6, (255, 255, 0), 1)
        # shorter, thinner crosshair
        cv2.line(overlay, (x - 15, y), (x + 15, y), (255, 255, 0), 1)
        cv2.line(overlay, (x, y - 15), (x, y + 15), (255, 255, 0), 1)

        # blend overlay back into img with low alpha
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

    def calculate_height_from_keypoints(self, kps, depth):
        return self.pose_detector.calculate_nose_height_3d(
            kps, depth, self.intr, self.depth_scale
        )

    def inference_loop(self):
        cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
        while self.running:
            frames  = self.pipeline.wait_for_frames()
            aligned = self.align.process(frames)
            cf = aligned.get_color_frame()
            df = aligned.get_depth_frame()
            if not cf or not df:
                continue

            img   = np.asanyarray(cf.get_data())
            depth = cv2.medianBlur(np.asanyarray(df.get_data()), 5)
            vis   = img.copy()

            self.draw_boundary(vis)
            self.draw_slam_point(vis)

            res = self.model.predict(
                img, conf=CONF_THRESH, iou=IOU_THRESH,
                classes=[0], verbose=False
            )[0]

            detection_objects = []
            if res.keypoints is not None:
                kps_data = res.keypoints.data.cpu().numpy()
                boxes    = res.boxes.data.cpu().numpy()

                # use zip so if either is empty, loop is skipped
                for kps, box in zip(kps_data, boxes):
                    x1, y1, x2, y2 = map(int, box[:4])
                    conf = float(box[4])

                    # Get ankle midpoint for better positioning
                    ankle_x, ankle_y, ankle_method = self.pose_detector.get_ankle_midpoint(kps)
                    
                    if ankle_x is None:
                        # Use nose-aligned bottom middle of bounding box as fallback
                        cx, cy, positioning_method = self.pose_detector.get_nose_aligned_bottom_fallback(kps, (x1, y1, x2, y2))
                    else:
                        cx, cy = ankle_x, ankle_y
                        positioning_method = ankle_method

                    if not point_in_polygon((cx,cy), self.boundary_points):
                        continue

                    patch = depth[max(0,cy-3):cy+4, max(0,cx-3):cx+4]
                    if patch.size == 0:
                        continue
                    z = float(np.median(patch)) * self.depth_scale
                    if z <= 0:
                        continue

                    Xc, Yc, _ = rs.rs2_deproject_pixel_to_point(self.intr, [cx,cy], z)
                    P = T_MAP_CAM @ np.array([Xc, Yc, z, 1.0], float)
                    Xm, Ym = float(P[0]), float(P[1])

                    height = self.calculate_height_from_keypoints(kps, depth)

                    detection_objects.append({
                        'x': Xm, 'y': Ym,
                        'height': height,
                        'confidence': conf,
                        'box': [x1, y1, x2, y2],
                        'keypoints': kps,
                        'bottom_center': (cx, cy),
                        'positioning_method': positioning_method
                    })

            nearest_det, nearest_idx = self.find_nearest_person(detection_objects)
            for idx, det in enumerate(detection_objects):
                x1, y1, x2, y2 = det['box']
                is_nearest     = (idx == nearest_idx)
                color          = (0,255,255) if is_nearest else (147,20,255)
                thickness      = 3          if is_nearest else 2

                cv2.rectangle(vis, (x1,y1), (x2,y2), color, thickness)
                cx, cy = det['bottom_center']
                
                # Different colors and styles for different positioning methods
                if det['positioning_method'] == "both_ankles":
                    circle_color = (0,255,0)  # Green for both ankles
                    circle_size = 6
                    outline_size = 8
                elif det['positioning_method'] in ["left_ankle", "right_ankle"]:
                    circle_color = (0,255,255)  # Yellow for single ankle
                    circle_size = 5
                    outline_size = 7
                elif det['positioning_method'] == "nose_aligned_bottom":
                    circle_color = (255,128,0)  # Orange for nose-aligned bottom
                    circle_size = 5
                    outline_size = 7
                else:
                    circle_color = (255,0,0)  # Blue for box center
                    circle_size = 4
                    outline_size = 6
                
                cv2.circle(vis, (cx,cy), circle_size, circle_color, -1)
                cv2.circle(vis, (cx,cy), outline_size, (255,255,255), 2)  # White outline
                
                # Add position indicator text
                pos_text = ""
                if det['positioning_method'] == "both_ankles":
                    pos_text = "MID"
                elif det['positioning_method'] == "left_ankle":
                    pos_text = "L"
                elif det['positioning_method'] == "right_ankle":
                    pos_text = "R"
                elif det['positioning_method'] == "nose_aligned_bottom":
                    pos_text = "NAB"
                else:
                    pos_text = "B"
                    
                cv2.putText(vis, pos_text, (cx-12, cy-12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 2)

                # Draw keypoints with special highlighting for ankles and nose
                for j,(kx,ky,kc) in enumerate(det['keypoints']):
                    if kc > 0.5:
                        if j == self.pose_detector.NOSE_IDX:
                            kcolor = (0,0,255)  # Red for nose
                        elif j in [self.pose_detector.LEFT_ANKLE_IDX, self.pose_detector.RIGHT_ANKLE_IDX]:
                            kcolor = (0,255,0)  # Green for ankles
                        else:
                            kcolor = (0,255,255)  # Yellow for other keypoints
                        cv2.circle(vis, (int(kx),int(ky)), 3, kcolor, -1)

                # Enhanced text information (more compact)
                txt = [
                    f"Conf: {det['confidence']:.2f}",
                    f"H: {det['height']:.2f}m" if det['height'] is not None else "H: N/A"
                ]
                if is_nearest:
                    dist = math.hypot(det['x']-self.slam_point[0], det['y']-self.slam_point[1])
                    txt += [f"D: {dist:.2f}m", "NEAREST"]
                draw_text_block(vis, txt, (x1+8, y1+8), 
                              font_scale=0.35, 
                              bg_color=(0,0,0,120), 
                              padding=4, 
                              line_spacing=12)

            # Create expanded canvas with modern header
            header_height = 100
            expanded_vis = np.zeros((vis.shape[0] + header_height, vis.shape[1], 3), dtype=np.uint8)
            expanded_vis[header_height:, :] = vis
            
            # Draw the modern header
            draw_modern_header(expanded_vis, self.slam_point, len(detection_objects), nearest_det, header_height)
            
            vis = expanded_vis

            with self.lock:
                self.latest_dets    = detection_objects
                self.nearest_person = nearest_det
                self.all_detections = detection_objects.copy()

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
            nearest = self.nearest_person

        people = []
        if nearest:
            pd = {
                "x": nearest['x'],
                "y": nearest['y'],
                "z": 0.0,
                "confidence": nearest['confidence'],
                "positioning_method": nearest['positioning_method']
            }
            if nearest['height'] is not None:
                pd["height"] = nearest['height']
            pd["distance_to_slam_point"] = math.hypot(
                nearest['x']-self.slam_point[0],
                nearest['y']-self.slam_point[1]
            )
            people.append(pd)

        msg = {
            "timestamp": time.time(),
            "frame_id": "map",
            "people": people,
            "slam_point": {"x": self.slam_point[0], "y": self.slam_point[1]},
            "total_in_boundary": len(self.all_detections)
        }
        try:
            self.mqtt.publish_human_results(json.dumps(msg))
            if nearest:
                method_info = f" ({nearest['positioning_method']})"
                print(f"[mqtt] Published nearest person: {nearest['x']:.2f}, {nearest['y']:.2f}, height: {nearest.get('height', 'N/A')}{method_info}")
            else:
                print(f"[mqtt] No people detected in boundary")
        except Exception as e:
            print("[mqtt error]", e)

    def stop(self):
        self.running = False
        self.pipeline.stop()
        cv2.destroyAllWindows()

def main():
    slam_point = parse_slam_point(sys.argv)
    pub = HumanPublisher(slam_point)
    try:
        print("Human detector started—press ESC to exit.")
        print("Position priority: Both Ankles > Single Ankle > Nose-Aligned Bottom > Box Center")
        while pub.running:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pub.stop()

if __name__ == '__main__':
    main()