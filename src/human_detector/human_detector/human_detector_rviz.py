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

class Human3DDetection(Node):
    def __init__(self):
        super().__init__('human_detection_rviz')

        self.declare_parameter('model_path')
        model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.get_logger().info(f'Loading YOLO detection model from: {model_path}')
        self.model = YOLO(model_path)

        self.bridge = CvBridge()
        self.depth_image = None
        self.depth_intrinsics = None

        self.create_subscription(Image, '/camera/realsense2_camera/color/image_raw', self.image_cb, 10)
        self.create_subscription(Image, '/camera/realsense2_camera/depth/image_rect_raw', self.depth_cb, 10)
        self.create_subscription(CameraInfo, '/camera/realsense2_camera/depth/camera_info', self.info_cb, 10)

        self.pub = self.create_publisher(Image, 'human_detector/detection_annotated', 10)
        self.marker_pub = self.create_publisher(Marker, '/human_detector/marker', 10)

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
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        results = self.model(frame, conf=0.6, iou=0.7)[0]
        annotated = frame.copy()

        for box in results.boxes:
            cls = int(box.cls[0])
            label = self.model.names[cls]
            if label not in ('person', 'teleco'):
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            # Clamp box to depth image boundaries
            x1c = max(x1, 0)
            y1c = max(y1, 0)
            x2c = min(x2, self.depth_image.shape[1] - 1)
            y2c = min(y2, self.depth_image.shape[0] - 1)

            if x2c <= x1c or y2c <= y1c:
                continue  # Invalid bbox

            patch = self.depth_image[y1c:y2c, x1c:x2c].astype(np.float32)
            valid_depths = patch[(patch > 0) & (patch < 10000)]  # in mm, clip outliers >10m

            if valid_depths.size == 0:
                continue

            z_mm = np.median(valid_depths)
            z = z_mm * 0.001  # to meters

            u = (x1 + x2) // 2
            v = (y1 + y2) // 2

            X, Y, Z = rs.rs2_deproject_pixel_to_point(self.depth_intrinsics, [u, v], z)
            dist_text = f"{Z:.2f} m"
            self.get_logger().info(f"[3D] {label} bbox ({x1},{y1})–({x2},{y2}) center ({u},{v}) @ Z={Z:.2f}m")

            # Marker for RViz
            marker = Marker()
            marker.header.frame_id = "camera_color_optical_frame"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "detections"
            marker.id = cls
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

            # Annotate image
            import cv2
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, f"{label} {dist_text}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

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
