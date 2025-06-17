import os

from launch import LaunchDescription
from launch.substitutions import LaunchConfiguration
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """
    Combined launch file that:
      1. Starts the RealSense camera (color, depth, and pointcloud)
      2. Publishes a static transform from camera_depth_optical_frame → base_link
      3. Launches YOLO (with use_3d=True) for 3D detections
      4. Opens RViz with a preconfigured .rviz file
    """

    # ─── 1) RealSense ===========

    # Locate the RealSense rs_launch.py in realsense2_camera
    realsense_pkg = get_package_share_directory("realsense2_camera")
    realsense_launch = os.path.join(realsense_pkg, "launch", "rs_launch.py")

    # Launch arguments for RealSense driver
    # Lower resolutions and frame rates reduce CPU/GPU load.
    realsense_args = {
        # Color stream: 640×480 @ 15 Hz
        "enable_color": "True",
        "color_width":  "640",
        "color_height": "480",
        "color_fps":    "15",

        # Depth stream: 640×480 @ 15 Hz
        "enable_depth": "True",
        "depth_width":  "320", # Reduced resolution for faster processing
        "depth_height": "320", # Reduced resolution for faster processing
        "depth_fps":    "10",

        # Enable + align pointcloud to color frame
        "pointcloud.enable": "True",
        "align_depth":       "True",
        # "pointcloud.enable": "False",
        # "align_depth":       "False",
    }
    include_realsense = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch),
        launch_arguments=realsense_args.items(),
    )

    # ─── 2) Static Transform ===========

    # Publish a static transform: camera_depth_optical_frame → base_link
    # Allows YOLO's 3D node to transform depth into "base_link".
    static_tf_node = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="camera_depth_to_base_link",
        arguments=[
            "0", "0", "0",  # x, y, z translation (none)
            "0", "0", "0",  # roll, pitch, yaw (none)
            "camera_depth_optical_frame",  # parent_frame
            "base_link"                     # child_frame
        ],
    )

    # ─── 3) YOLO (3D) ===========

    # Path to the YOLO launch file (yolo.launch.py) in yolo_bringup
    yolo_pkg = get_package_share_directory("yolo_bringup")
    yolo_launch_file = os.path.join(yolo_pkg, "launch", "yolo.launch.py")

    # YOLO launch arguments
    #   • use_3d = True → run detect_3d_node
    #   • reduce inference size to 320×320
    #   • disable augmentation / heavy settings to cut lag
    yolo_args = {
        # Boolean flags
        "use_tracking":   LaunchConfiguration("use_tracking", default="True"),
        "use_3d":         LaunchConfiguration("use_3d",       default="False"),

        # Model / tracker / device
        "model_type":     LaunchConfiguration("model_type", default="YOLO"),
        "model":          LaunchConfiguration("model",      default="/home/commu/Desktop/human_detector_ws/models/best_yolo11s.pt"),
        "tracker":        LaunchConfiguration("tracker",    default="bytetrack.yaml"),
        "device":         LaunchConfiguration("device",     default="cuda:0"),   # Change to "cuda:0" if GPU is available
        "enable":         LaunchConfiguration("enable",     default="True"),

        # Detection thresholds
        "threshold":      LaunchConfiguration("threshold", default="0.5"),
        "iou":            LaunchConfiguration("iou",       default="0.7"),

        # Inference image size (smaller → faster)
        "imgsz_height":   LaunchConfiguration("imgsz_height", default="256"),
        "imgsz_width":    LaunchConfiguration("imgsz_width",  default="256"),

        # Precision / max detections / augment
        "half":           LaunchConfiguration("half",       default="True"),  # Set "True" if you have FP16 support on GPU
        "max_det":        LaunchConfiguration("max_det",    default="50"),    # reduce max detections
        "augment":        LaunchConfiguration("augment",    default="False"),
        "agnostic_nms":   LaunchConfiguration("agnostic_nms", default="False"),
        "retina_masks":   LaunchConfiguration("retina_masks", default="False"),

        # Color image topic + QoS
        "input_image_topic": LaunchConfiguration(
                                "input_image_topic",
                                default="/camera/camera/color/image_raw"
                             ),
        "image_reliability": LaunchConfiguration("image_reliability", default="0"),

        # Depth image topic + QoS
        "input_depth_topic": LaunchConfiguration(
                                "input_depth_topic",
                                default="/camera/camera/depth/image_rect_raw"
                             ),
        "depth_image_reliability": LaunchConfiguration("depth_image_reliability", default="0"),
        "input_depth_info_topic": LaunchConfiguration(
                                     "input_depth_info_topic",
                                     default="/camera/camera/depth/camera_info"
                                  ),
        "depth_info_reliability": LaunchConfiguration("depth_info_reliability", default="0"),

        # 3D parameters (target_frame must match static_tf_node child_frame)
        "target_frame":               LaunchConfiguration("target_frame", default="base_link"),
        "depth_image_units_divisor":  LaunchConfiguration("depth_image_units_divisor", default="1000"),
        "maximum_detection_threshold": LaunchConfiguration("maximum_detection_threshold", default="0.3"),

        # Namespace + debug
        "namespace":   LaunchConfiguration("namespace", default="yolo"),
        "use_debug":   LaunchConfiguration("use_debug", default="True"),
    }
    include_yolo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(yolo_launch_file),
        launch_arguments=yolo_args.items(),
    )

    # ─── 4) RViz ===========

    # Launch RViz with a saved configuration (optional).
    # If you don’t have a .rviz file, remove the "arguments" field and configure RViz manually.
    rviz_config = os.path.join(
        get_package_share_directory("yolo_bringup"),
        "launch",
        "yolo_3d.rviz"
    )
    rviz_node = Node(
        package="rviz2",
        executable="rviz2",
        name="rviz2",
        output="screen",
        arguments=["-d", '/home/commu/Desktop/human_detector_ws/src/yolo_ros/yolo_bringup/launch/yolo_3d.rviz']
    )

    # ─── Final LaunchDescription ===========

    return LaunchDescription([
        include_realsense,
        static_tf_node,
        include_yolo,
        rviz_node,
    ])


