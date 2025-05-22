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
        default_model = f'{share_dir}/models/yolov11n.pt'  # your detection-only checkpoint
        self.declare_parameter('model_path', default_model)
        model_path = self.get_parameter('model_path')\
                             .get_parameter_value().string_value
        self.get_logger().info(f'Loading YOLO detection model from: {model_path}')
        self.model = YOLO(model_path)

        # --- OpenCV bridge & window ---
        self.bridge = CvBridge()
        cv2.namedWindow('Detection', cv2.WINDOW_NORMAL)

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
        # raw depth in millimeters
        self.depth_image = self.bridge.imgmsg_to_cv2(
            msg, desired_encoding='passthrough'
        )

    def info_cb(self, msg: CameraInfo):
        # fill pyrealsense2 intrinsics
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
        # 1) convert color frame
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # 2) run detection
        results = self.model(frame)[0]

        # 3) copy for drawing
        annotated = frame.copy()

        # 4) loop through detections
        for box in results.boxes:
            cls   = int(box.cls[0])
            label = self.model.names[cls]
            if label != 'person':
                continue

            # get box coords
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

            # draw 2D box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)

            # compute center pixel
            u = (x1 + x2) // 2
            v = (y1 + y2) // 2

            dist_text = "n/a"
            if (self.depth_image is not None
               and self.depth_intrinsics is not None
               and 0 <= v < self.depth_image.shape[0]
               and 0 <= u < self.depth_image.shape[1]):

                z_mm = float(self.depth_image[v, u])
                z = z_mm * 0.001  # to meters
                if z > 0:
                    # deproject to 3D
                    X, Y, Z = rs.rs2_deproject_pixel_to_point(
                        self.depth_intrinsics, [u, v], z
                    )
                    self.get_logger().info(
                        f"[3D] Person at pixel ({u},{v}): "
                        f"X={X:.2f}m  Y={Y:.2f}m  Z={Z:.2f}m"
                    )
                    dist_text = f"{Z:.2f} m"

            # --- neat background + text overlay ---
            dist_label = f"{label} {dist_text}"
            font       = cv2.FONT_HERSHEY_SIMPLEX
            scale      = 0.5
            thk        = 1

            # measure text size
            (w_text, h_text), baseline = cv2.getTextSize(
                dist_label, font, scale, thk
            )
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

            # draw filled background (black)
            cv2.rectangle(
                annotated,
                (x_bg1, y_bg1),
                (x_bg2, y_bg2),
                (0, 0, 0),
                cv2.FILLED
            )

            # draw the text (green)
            cv2.putText(
                annotated,
                dist_label,
                (x_bg1 + pad, y_bg2 - baseline - pad // 2),
                font,
                scale,
                (0, 255, 0),
                thk,
                cv2.LINE_AA
            )

        # 5) show & publish
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
