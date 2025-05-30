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

class Human2DDistance(Node):
    def __init__(self):
        super().__init__('human_detection')

        # --- load detection model ---
        share_dir     = get_package_share_directory('human_detector')
        # default_model = f'{share_dir}/models/best.pt'
        # self.declare_parameter('model_path', default_model)

        self.declare_parameter('model_path') # Expect this to be set in the launch file
        model_path = self.get_parameter('model_path')\
                             .get_parameter_value().string_value
        self.get_logger().info(f'Loading YOLO detection model from: {model_path}')
        self.model = YOLO(model_path)

        # --- OpenCV bridge & window ---
        self.bridge = CvBridge()
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Detection', 1280, 720)  # or any desired resolution

        # --- depth containers ---
        self.depth_image = None
        self.depth_intrinsics = None

        # --- subscriptions ---
        self.create_subscription(
            Image,
            '/camera/realsense2_camera/color/image_raw',
            self.image_cb, 10)
        self.create_subscription(
            Image,
            '/camera/realsense2_camera/depth/image_rect_raw',
            self.depth_cb, 10)
        self.create_subscription(
            CameraInfo,
            '/camera/realsense2_camera/depth/camera_info',
            self.info_cb, 10)

        # --- republish annotated image ---
        self.pub = self.create_publisher(
            Image,
            'human_detector/detection_annotated',
            10
        )

    def depth_cb(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='passthrough'
        )

    def info_cb(self, msg: CameraInfo):
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
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        # Run YOLO inference with confidence and IoU thresholds:
        #   - conf (default 0.25): minimum confidence score required to keep a detection (0.0–1.0)
        #           higher = fewer false positives, lower = more detections (but noisier)
        #   - iou (default 0.7):  intersection-over-union threshold for non-max suppression (0.0–1.0)
        #           lower = removes more overlapping boxes, higher = keeps more overlaps
        #   - [0]:  selects the first (and only) result in the batch
        # results = self.model(frame)[0]
        results = self.model(frame, conf=0.6, iou=0.7)[0]


        annotated = frame.copy()

        for box in results.boxes:
            cls   = int(box.cls[0])
            label = self.model.names[cls]

            # <<--- REMOVE this if you want **all** classes drawn:
            # if label != 'person':
            #     continue

            # OR if you only want person+teleco:
            if label not in ('person','teleco'):
                continue

            # draw 2D box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)

            # compute center pixel for distance
            u = (x1 + x2) // 2
            v = (y1 + y2) // 2

            dist_text = "n/a"
            if (self.depth_image is not None
               and self.depth_intrinsics is not None
               and 0 <= v < self.depth_image.shape[0]
               and 0 <= u < self.depth_image.shape[1]):

                z_mm = float(self.depth_image[v, u])
                z = z_mm * 0.001  # meters
                if z > 0:
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(
                        self.depth_intrinsics, [u, v], z
                    )
                    dist_text = f"{Z:.2f} m"
                    self.get_logger().info(
                        f"[3D] {label} at ({u},{v}): X={X:.2f}m Y={Y:.2f}m Z={Z:.2f}m"
                    )

            # draw background & label+distance
            dist_label = f"{label} {dist_text}"
            font, scale, thk = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            (w_text, h_text), baseline = cv2.getTextSize(dist_label, font, scale, thk)
            pad = 4
            x_bg1, y_bg1 = x1, y1 - h_text - baseline - pad
            x_bg2, y_bg2 = x1 + w_text + pad*2, y1
            if y_bg1 < 0:
                y_bg1, y_bg2 = y1, y1 + h_text + baseline + pad
            cv2.rectangle(annotated, (x_bg1,y_bg1), (x_bg2,y_bg2), (0,0,0), cv2.FILLED)
            cv2.putText(annotated, dist_label,
                        (x_bg1+pad, y_bg2-baseline-pad//2),
                        font, scale, (0,255,0), thk, cv2.LINE_AA)

        cv2.imshow('Detection', annotated)
        cv2.waitKey(1)
        out = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out.header = msg.header
        self.pub.publish(out)

def main(args=None):
    rclpy.init(args=args)
    node = Human2DDistance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()