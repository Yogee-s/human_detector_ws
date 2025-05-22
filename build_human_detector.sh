#!/usr/bin/env bash
# run ./build_human_detector.sh
# -------------------------------------------------------------------


# 1) Define workspace root
WORKSPACE=~/Desktop/human_detector_ws

# 2) Clear old build/install/log
echo "Removing old build, install, and log directories..."
rm -rf "$WORKSPACE"/build "$WORKSPACE"/install "$WORKSPACE"/log

# 3) Source system ROS 2 Humble
echo "Sourcing ROS 2 Humble..."
source /opt/ros/humble/setup.bash

# 4) Build workspace with merge-install
echo "Building workspace (merge-install)..."
cd "$WORKSPACE" || exit 1
colcon build --merge-install

# 5) Source workspace overlay
echo "Sourcing workspace overlay..."
source "$WORKSPACE"/install/setup.bash

# 6) Prepend workspace to ROS env vars
echo "Exporting ROS environment variables..."
export AMENT_PREFIX_PATH="$WORKSPACE"/install:$AMENT_PREFIX_PATH
export CMAKE_PREFIX_PATH="$WORKSPACE"/install:$CMAKE_PREFIX_PATH
export ROS_PACKAGE_PATH="$WORKSPACE"/install/share:$ROS_PACKAGE_PATH

# 7) Done
echo
echo "=== Setup Complete ==="
echo "AMENT_PREFIX_PATH = $AMENT_PREFIX_PATH"
echo "CMAKE_PREFIX_PATH = $CMAKE_PREFIX_PATH"
echo "ROS_PACKAGE_PATH  = $ROS_PACKAGE_PATH"
echo
echo "You can now run:"
echo "  ros2 pkg list | grep human_detector"
echo "  ros2 launch human_detector human_detector_launch.py"

