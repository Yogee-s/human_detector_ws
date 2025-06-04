import sys
import rclpy
import numpy as np
import pyrealsense2 as rs
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from visualization_msgs.msg import Marker
from cv_bridge import CvBridge
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
import cv2

class Human3DDetection(Node):
    def __init__(self):
        super().__init__('human_detection_rviz')

        # Load YOLO model path parameter
        self.declare_parameter('model_path')
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.get_logger().info(f'Loading YOLO detection model from: {model_path}')
        self.model = YOLO(model_path)

        # CV Bridge and RealSense storage
        self.bridge = CvBridge()
        self.depth_image = None
        self.depth_intrinsics = None

        # Subscriptions
        self.create_subscription(Image, '/camera/realsense2_camera/color/image_raw', self.image_cb, 10)
        self.create_subscription(Image, '/camera/realsense2_camera/depth/image_rect_raw', self.depth_cb, 10)
        self.create_subscription(CameraInfo, '/camera/realsense2_camera/depth/camera_info', self.info_cb, 10)

        # Publishers
        self.pub = self.create_publisher(Image, 'human_detector/detection_annotated', 10)
        self.marker_pub = self.create_publisher(Marker, '/human_detector/marker', 10)

        # Smoothing: EMA factor and track storage
        self.alpha = 0.7
        self.tracks = {}

    def depth_cb(self, msg: Image):
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def info_cb(self, msg: CameraInfo):
        intr = rs.intrinsics()
        intr.width = msg.width
        intr.height = msg.height
        intr.fx = msg.k[0]
        intr.fy = msg.k[4]
        intr.ppx = msg.k[2]
        intr.ppy = msg.k[5]
        intr.model = rs.distortion.brown_conrady
        intr.coeffs = list(msg.d)
        self.depth_intrinsics = intr

    def image_cb(self, msg: Image):
        if self.depth_image is None or self.depth_intrinsics is None:
            return

        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model.track(
            frame, conf=0.7, iou=0.6, tracker='bytetrack.yaml', persist=True
        )[0]

        annotated = frame.copy()
        current_ids = set()

        # Define colors (BGR)
        box_color = (255, 128, 0)            # sky blue box
        text_bg_color = (0, 128, 255)        # vibrant orange background
        text_color = (255, 255, 255)         # white text

        for box in results.boxes:
            cls = int(box.cls[0])
            raw_id = box.id if hasattr(box, 'id') else None
            track_id = int(raw_id[0]) if raw_id is not None else None
            if track_id is None:
                continue
            current_ids.add(track_id)

            label = self.model.names[cls]
            if label not in ('person', 'teleco'):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            x1c, y1c = max(x1, 0), max(y1, 0)
            x2c = min(x2, self.depth_image.shape[1] - 1)
            y2c = min(y2, self.depth_image.shape[0] - 1)
            if x2c <= x1c or y2c <= y1c:
                continue

            patch = self.depth_image[y1c:y2c, x1c:x2c].astype(np.float32)
            valid = patch[(patch > 0) & (patch < 10000)]
            if valid.size == 0:
                continue
            z = float(np.median(valid)) * 0.001

            u, v = (x1 + x2)//2, (y1 + y2)//2
            X, Y, Z = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [u, v], z)

            if track_id in self.tracks:
                prev = self.tracks[track_id]
                x1 = int(self.alpha*x1 + (1-self.alpha)*prev['x1'])
                y1 = int(self.alpha*y1 + (1-self.alpha)*prev['y1'])
                x2 = int(self.alpha*x2 + (1-self.alpha)*prev['x2'])
                y2 = int(self.alpha*y2 + (1-self.alpha)*prev['y2'])
                Z  =     self.alpha*Z  + (1-self.alpha)*prev['z']
            self.tracks[track_id] = {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'z': Z}

            # RViz marker
            marker = Marker()
            marker.header.frame_id = 'camera_color_optical_frame'
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'detections'
            marker.id = track_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = X
            marker.pose.position.y = Y
            marker.pose.position.z = Z
            marker.scale.x = marker.scale.y = marker.scale.z = 0.1
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 0.8
            self.marker_pub.publish(marker)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

            dist_text = f"{label} {Z:.2f}m"
            (tw, th), _ = cv2.getTextSize(dist_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            bg_tl = (x1, y1)
            bg_br = (x1 + tw + 6, y1 + th + 6)
            cv2.rectangle(annotated, bg_tl, bg_br, text_bg_color, cv2.FILLED)
            cv2.putText(annotated, dist_text, (x1 + 3, y1 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        # Remove stale markers
        stale_ids = set(self.tracks.keys()) - current_ids
        for sid in stale_ids:
            d = Marker()
            d.header.frame_id = 'camera_color_optical_frame'
            d.header.stamp = self.get_clock().now().to_msg()
            d.ns = 'detections'
            d.id = sid
            d.action = Marker.DELETE
            self.marker_pub.publish(d)
            del self.tracks[sid]

        out = self.bridge.cv2_to_imgmsg(annotated, 'bgr8')
        out.header = msg.header
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = Human3DDetection()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()