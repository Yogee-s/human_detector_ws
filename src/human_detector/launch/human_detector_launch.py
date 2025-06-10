from launch import LaunchDescription
from launch_ros.actions import Node

from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    return LaunchDescription([
        # RealSense driver: leave out serial_no to auto-select the first camera
        Node(
            package='realsense2_camera',
            executable='realsense2_camera_node',
            name='realsense2_camera',
            output='screen',
            parameters=[{
                # no serial_no → will use the first available device
                'depth_module.profile': '640x480x30',
                'color_module.profile': '640x480x30',
                'pointcloud.enable': True,
                'align_depth.enable': True,
                'unite_imu_method': 'none',
                'publish_odom_tf': False,
                'allow_no_texture_points': True,
                'clip_distance': 3.0,
            }]
        ),

        # Human detection node
        # Node(
        #     package='human_detector',
        #     executable='human_detection',
        #     name='human_detection',
        #     output='screen',
        #     parameters=[{
        #         'model_path': '../../models/best.pt',
        #     }]
        # ),

        # Human pose node
        # Node(
        #     package='human_detector',
        #     executable='human_pose',
        #     name='human_pose',
        #     output='screen',
        #     parameters=[{
        #         # 'model_path': '../../models/yolov8n-pose.pt',
        #         'model_path': '../../models/yolov11s-pose.pt',
        #     }]
        # ),

        # Human detection node with RViz visualization
        Node(
            package='human_detector',
            executable='human_detection_rviz',
            name='human_detection_rviz',
            output='screen',
            parameters=[{
                'model_path': '/home/commu/Desktop/human_detector_ws/models/best_yolo11s.pt',
            }]
        ),

        # Node(
        #     package='rqt_image_view',
        #     executable='rqt_image_view',
        #     name='rqt_image_view',
        #     output='screen'
        # ),

        # RViz2 visualizer
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', '/home/commu/Desktop/human_detector_ws/src/human_detector/launch/human_detection.rviz'],
            output='screen'
        ),
 
    ])

