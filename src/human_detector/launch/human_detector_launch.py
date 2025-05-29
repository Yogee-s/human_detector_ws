from launch import LaunchDescription
from launch_ros.actions import Node

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
            }]
        ),
        # Human detection node
        Node(
            package='human_detector',
            executable='human_detection',
            name='human_detection',
            output='screen',
            parameters=[{
                'model_path': '/home/yogee/Desktop/human_detector_ws/src/human_detector/models/best_v2_yolov11n.pt',
            }]
        ),

        # Human pose node
        # Node(
        #     package='human_detector',
        #     executable='human_pose',
        #     name='human_pose',
        #     output='screen',
        #     parameters=[{
        #         # 'model_path': '/home/yogee/Desktop/human_detector_ws/src/human_detector/models/yolov8n-pose.pt',
        #         'model_path': '/home/yogee/Desktop/human_detector_ws/src/human_detector/models/yolov11s-pose.pt',
        #     }]
        # ),


        # RViz2 visualizer
        # Node(
        #     package='rviz2',
        #     executable='rviz2',
        #     name='rviz2',
        #     arguments=['-d', 'human_detection.rviz'],
        #     output='screen'
        # ),
 
    ])

