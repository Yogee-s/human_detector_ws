from setuptools import setup

package_name = 'human_detector'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        # 1) register package in the ament index
        ('share/ament_index/resource_index/packages', 
         ['resource/human_detector']),
        # 2) install package.xml
        ('share/human_detector', ['package.xml']),
        # 3) install launch file
        ('share/human_detector/launch', ['launch/human_detector_launch.py']),
    ],
    install_requires=[
        'ultralytics>=8.0.0',
        'opencv-python',
        'pyrealsense2',
        'numpy'
    ],
    package_data={package_name: ['models/*.pt']},
    entry_points={
        'console_scripts': [
            'human_pose = human_detector.human_pose_node:main',
            'human_detection = human_detector.human_detection_node:main',
        ],
    },
)

