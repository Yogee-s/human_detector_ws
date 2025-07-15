#!/usr/bin/env python3
import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import math

# === CONFIGURATION: edit these only ===
MODEL_PATH     = 'models/yolov11n-pose.pt'  # Changed to pose model
CONF_THRESH    = 0.4      # confidence threshold
IOU_THRESH     = 0.7      # NMS IoU threshold
FRAME_WIDTH    = 640      # RealSense color/depth width
FRAME_HEIGHT   = 480      # RealSense color/depth height
USE_TRACKING   = True     # False = detect only, True = track()
TRACKER_CONFIG = 'bytetrack.yaml'
# =====================================

class PoseHeightDetector:
    def __init__(self):
        # COCO pose keypoints (17 keypoints for human pose)
        self.keypoint_names = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]
        
        # Indices for key points
        self.NOSE_IDX = 0
        self.LEFT_ANKLE_IDX = 15
        self.RIGHT_ANKLE_IDX = 16
        self.LEFT_KNEE_IDX = 13
        self.RIGHT_KNEE_IDX = 14
        
    def get_ground_level(self, keypoints):
        """
        Estimate ground level from ankle positions
        """
        # Check if keypoints array is valid
        if keypoints.shape[0] <= max(self.LEFT_ANKLE_IDX, self.RIGHT_ANKLE_IDX):
            return None
            
        left_ankle = keypoints[self.LEFT_ANKLE_IDX]
        right_ankle = keypoints[self.RIGHT_ANKLE_IDX]
        
        # Use the lowest visible ankle as ground reference
        ground_candidates = []
        
        if len(left_ankle) >= 3 and left_ankle[2] > 0.5:  # confidence threshold
            ground_candidates.append(left_ankle[1])
        if len(right_ankle) >= 3 and right_ankle[2] > 0.5:
            ground_candidates.append(right_ankle[1])
            
        if ground_candidates:
            return max(ground_candidates)  # Lower in image = higher Y value
        else:
            return None
    
    def calculate_nose_height_pixels(self, keypoints):
        """
        Calculate height from ground to nose in pixels
        """
        # Check if keypoints array is valid
        if keypoints.shape[0] <= self.NOSE_IDX:
            return None, None
            
        nose = keypoints[self.NOSE_IDX]
        
        # Check if nose is visible
        if len(nose) < 3 or nose[2] < 0.5:  # confidence threshold
            return None, None
            
        ground_y = self.get_ground_level(keypoints)
        if ground_y is None:
            return None, None
            
        # Calculate height (ground_y - nose_y because Y increases downward)
        height_pixels = ground_y - nose[1]
        
        return height_pixels if height_pixels > 0 else None, ground_y
    
    def calculate_nose_height_3d(self, keypoints, depth_image, depth_intrinsics, depth_scale):
        """
        Calculate real-world height from ground to nose using depth data
        """
        # Check if keypoints array is valid
        if keypoints.shape[0] <= self.NOSE_IDX:
            return None
            
        nose = keypoints[self.NOSE_IDX]
        
        # Check if nose is visible
        if len(nose) < 3 or nose[2] < 0.5:
            return None
            
        # Get ankle positions for ground reference
        left_ankle = keypoints[self.LEFT_ANKLE_IDX]
        right_ankle = keypoints[self.RIGHT_ANKLE_IDX]
        
        # Get 3D coordinates for nose
        nose_x, nose_y = int(nose[0]), int(nose[1])
        nose_depth = self.get_depth_at_point(depth_image, nose_x, nose_y) * depth_scale
        if nose_depth <= 0:
            return None
            
        nose_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [nose_x, nose_y], nose_depth)
        
        # Get 3D coordinates for ground reference (ankles)
        ground_3d_points = []
        
        if (keypoints.shape[0] > self.LEFT_ANKLE_IDX and 
            len(keypoints[self.LEFT_ANKLE_IDX]) >= 3 and 
            keypoints[self.LEFT_ANKLE_IDX][2] > 0.5):
            left_ankle = keypoints[self.LEFT_ANKLE_IDX]
            ankle_x, ankle_y = int(left_ankle[0]), int(left_ankle[1])
            ankle_depth = self.get_depth_at_point(depth_image, ankle_x, ankle_y) * depth_scale
            if ankle_depth > 0:
                ankle_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [ankle_x, ankle_y], ankle_depth)
                ground_3d_points.append(ankle_3d)
        
        if (keypoints.shape[0] > self.RIGHT_ANKLE_IDX and 
            len(keypoints[self.RIGHT_ANKLE_IDX]) >= 3 and 
            keypoints[self.RIGHT_ANKLE_IDX][2] > 0.5):
            right_ankle = keypoints[self.RIGHT_ANKLE_IDX]
            ankle_x, ankle_y = int(right_ankle[0]), int(right_ankle[1])
            ankle_depth = self.get_depth_at_point(depth_image, ankle_x, ankle_y) * depth_scale
            if ankle_depth > 0:
                ankle_3d = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [ankle_x, ankle_y], ankle_depth)
                ground_3d_points.append(ankle_3d)
        
        if not ground_3d_points:
            return None
            
        # Use average Y coordinate of ankles as ground level
        avg_ground_y = sum(point[1] for point in ground_3d_points) / len(ground_3d_points)
        
        # Calculate height (nose Y - ground Y, but Y is negative upward in camera frame)
        height_3d = avg_ground_y - nose_3d[1]  # This gives positive height
        
        return height_3d if height_3d > 0 else None
    
    def get_depth_at_point(self, depth_image, x, y, patch_size=5):
        """
        Get median depth value at a point using a small patch
        """
        half = patch_size // 2
        h, w = depth_image.shape
        
        y_min, y_max = max(0, y - half), min(h, y + half + 1)
        x_min, x_max = max(0, x - half), min(w, x + half + 1)
        
        patch = depth_image[y_min:y_max, x_min:x_max]
        if patch.size == 0:
            return 0
        return np.median(patch)

def main():
    # Initialize pose model (only detects person class)
    model = YOLO(MODEL_PATH)
    pose_detector = PoseHeightDetector()

    # Colors for visualization
    class_colors = {
        'person': (255, 0, 255),  # purple
    }
    keypoint_color = (0, 255, 0)  # green
    nose_color = (0, 0, 255)  # red
    ground_color = (255, 255, 0)  # cyan

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

            # detect or track poses - only person class
            if USE_TRACKING:
                results = model.track(
                    img, conf=CONF_THRESH, iou=IOU_THRESH,
                    tracker=TRACKER_CONFIG, persist=True,
                    classes=[0],  # Only detect class 0 (person)
                )[0]
            else:
                results = model(img, conf=CONF_THRESH, iou=IOU_THRESH, classes=[0])[0]

            # Process pose detections
            if results.keypoints is not None and len(results.keypoints.data) > 0:
                keypoints_data = results.keypoints.data.cpu().numpy()
                boxes = results.boxes.data.cpu().numpy() if results.boxes is not None else None
                
                # Check if we have valid detections
                if len(keypoints_data) == 0:
                    continue
                    
                for i, keypoints in enumerate(keypoints_data):
                    # Get bounding box info
                    if boxes is not None and len(boxes) > i:
                        x1, y1, x2, y2 = map(int, boxes[i][:4])
                        confidence = float(boxes[i][4])
                        
                        # Draw bounding box
                        cv2.rectangle(img, (x1, y1), (x2, y2), class_colors['person'], 2)
                        
                        # Class label
                        cv2.putText(img, f'Person {confidence:.2f}', 
                                   (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.6, class_colors['person'], 2)
                    
                    # Draw keypoints
                    for j, (x, y, conf) in enumerate(keypoints):
                        if j >= len(keypoints) or len(keypoints[j]) < 3:
                            continue
                        if conf > 0.5:
                            color = nose_color if j == pose_detector.NOSE_IDX else keypoint_color
                            cv2.circle(img, (int(x), int(y)), 3, color, -1)
                            
                            # Label nose
                            if j == pose_detector.NOSE_IDX:
                                cv2.putText(img, 'NOSE', (int(x) + 5, int(y) - 5),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, nose_color, 1)
                    
                    # Calculate heights
                    height_pixels, ground_y = pose_detector.calculate_nose_height_pixels(keypoints)
                    height_3d = pose_detector.calculate_nose_height_3d(
                        keypoints, depth_image, depth_intrinsics, depth_scale
                    )
                    
                    # Draw height measurements
                    if height_pixels is not None and ground_y is not None:
                        # Check if nose keypoint is valid
                        if (len(keypoints) > pose_detector.NOSE_IDX and 
                            len(keypoints[pose_detector.NOSE_IDX]) >= 3):
                            nose_pos = keypoints[pose_detector.NOSE_IDX]
                            nose_x, nose_y = int(nose_pos[0]), int(nose_pos[1])
                            
                            # Draw height line
                            cv2.line(img, (nose_x, int(ground_y)), (nose_x, nose_y), 
                                    ground_color, 2)
                            
                            # Height text
                            height_text = f"H: {height_pixels:.1f}px"
                            if height_3d is not None:
                                height_text += f" | {height_3d:.2f}m"
                            
                            cv2.putText(img, height_text, (nose_x + 10, nose_y - 20),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, ground_color, 2)
                            
                            # Print to console for robot control
                            print(f"Person {i+1}: Nose Height = {height_pixels:.1f} pixels", end="")
                            if height_3d is not None:
                                print(f" | {height_3d:.2f} meters", end="")
                            print(f" | Nose pos: ({nose_x}, {nose_y})")
                    
                    # Draw ground reference line
                    if ground_y is not None:
                        cv2.line(img, (0, int(ground_y)), (FRAME_WIDTH, int(ground_y)), 
                                ground_color, 1)
                        cv2.putText(img, "GROUND", (10, int(ground_y) - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, ground_color, 1)

            cv2.imshow('RealSense Pose Height Detection', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()