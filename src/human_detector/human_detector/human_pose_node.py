#!/usr/bin/env python3

import sys
import rclpy
import cv2
import numpy as np
import pyrealsense2 as rs
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory

class HumanDetector(Node):
    def __init__(self):
        super().__init__('human_detector')

        # 1) Load YOLOv8-Pose
        share_dir     = get_package_share_directory('human_detector')
        # default_model = f'{share_dir}/models/yolov8n-pose.pt'
        default_model = f'{share_dir}/models/yolov11s-pose.pt'
        self.declare_parameter('model_path', default_model)
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.get_logger().info(f'Loading model: {model_path}')
        self.model = YOLO(model_path)

        # 2) OpenCV + CvBridge
        self.bridge = CvBridge()
        cv2.namedWindow('Pose', cv2.WINDOW_NORMAL)

        # 3) Depth cache
        self.depth_image = None
        self.depth_intrinsics = None

        # 4) Subscriptions
        self.create_subscription(
            Image,
            '/camera/realsense2_camera/color/image_raw',
            self.image_cb, 10
        )
        self.create_subscription(
            Image,
            '/camera/realsense2_camera/depth/image_rect_raw',
            self.depth_cb, 10
        )
        self.create_subscription(
            CameraInfo,
            '/camera/realsense2_camera/depth/camera_info',
            self.depth_info_cb, 10
        )

        # 5) Republish annotated image
        self.pub = self.create_publisher(Image, 'human_detector/pose_annotated', 10)

    def depth_cb(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def depth_info_cb(self, msg: CameraInfo):
        intr = rs.intrinsics()
        intr.width  = msg.width
        intr.height = msg.height
        intr.fx     = msg.k[0]
        intr.fy     = msg.k[4]
        intr.ppx    = msg.k[2]
        intr.ppy    = msg.k[5]
        intr.model  = rs.distortion.brown_conrady
        intr.coeffs = list(msg.d)
        self.depth_intrinsics = intr

    def image_cb(self, msg: Image):
        # A) Convert color
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # B) Inference
        results = self.model(frame)[0]

        # C) Copy
        annotated = frame.copy()

        # D) If no detections, just show blank and exit early
        if len(results.boxes) == 0:
            cv2.imshow('Pose', annotated)
            cv2.waitKey(1)
            out = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
            out.header = msg.header
            self.pub.publish(out)
            return

        # E) Safely extract keypoints if present
        kp_xy = None
        kp_conf = None
        kp_obj = getattr(results, 'keypoints', None)
        if kp_obj is not None and kp_obj.xy is not None and kp_obj.conf is not None:
            if kp_obj.xy.shape[0] > 0:
                kp_xy   = kp_obj.xy.cpu().numpy()      # shape [n_det, n_kpt, 2]
                kp_conf = kp_obj.conf.cpu().numpy()    # shape [n_det, n_kpt]

        # F) Loop detections
        for i, box in enumerate(results.boxes):
            cls   = int(box.cls[0])
            label = self.model.names[cls]
            if label != 'person':
                continue

            # 1) Draw 2D box + label
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # 2) Draw pose keypoints
            if kp_xy is not None:
                for (x, y), c in zip(kp_xy[i], kp_conf[i]):
                    if c > 0.3:
                        cv2.circle(annotated, (int(x), int(y)), 3, (0, 0, 255), -1)

            # 3) Compute & draw 3D distance at the box center
            if self.depth_image is not None and self.depth_intrinsics is not None:
                u = (x1 + x2) // 2
                v = (y1 + y2) // 2
                z_mm = int(self.depth_image[v, u])
                z = z_mm * 0.001  # in meters
                if z > 0:
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(
                        self.depth_intrinsics, [u, v], z
                    )
                    # log it
                    self.get_logger().info(
                        f"[3D] Person#{i}: X={X:.2f}m  Y={Y:.2f}m  Z={Z:.2f}m"
                    )

                    # --- neat background + text overlay ---
                    dist_text = f"{label} {Z:.2f} m"
                    font     = cv2.FONT_HERSHEY_SIMPLEX
                    scale    = 0.5
                    thk      = 1

                    # measure text size
                    (w_text, h_text), baseline = cv2.getTextSize(dist_text, font, scale, thk)
                    pad = 4

                    # background rectangle coords
                    x_bg1 = x1
                    y_bg1 = y1 - h_text - baseline - pad
                    x_bg2 = x1 + w_text + pad * 2
                    y_bg2 = y1

                    # if it would go off the top, draw below instead
                    if y_bg1 < 0:
                        y_bg1 = y1
                        y_bg2 = y1 + h_text + baseline + pad

                    # draw filled background
                    cv2.rectangle(
                        annotated,
                        (x_bg1, y_bg1),
                        (x_bg2, y_bg2),
                        (0, 0, 0),      # black background
                        cv2.FILLED
                    )

                    # draw the text
                    cv2.putText(
                        annotated,
                        dist_text,
                        (x_bg1 + pad, y_bg2 - baseline - pad//2),
                        font,
                        scale,
                        (0, 255, 0),    # green text
                        thk,
                        cv2.LINE_AA
                    )


        # G) Display & publish
        cv2.imshow('Pose', annotated)
        cv2.waitKey(1)

        out = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out.header = msg.header
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = HumanDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
