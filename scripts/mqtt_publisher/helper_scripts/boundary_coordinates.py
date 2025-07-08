#!/usr/bin/env python3
import cv2
import numpy as np
import pyrealsense2 as rs
import time

# Configuration - match your main script
FRAME_W, FRAME_H = 640, 480

# Camera → map transform (same as your main script)
T_MAP_CAM = np.array([
    [0.09589393, -0.45845295, 0.88352999, -4.00730774],
    [-0.98829735, 0.06193290, 0.13940109, 0.54129092],
    [-0.11862842, -0.88655807, -0.44714885, 1.63494930],
    [0.00000000, 0.00000000, 0.00000000, 1.00000000],
], dtype=float)

class BoundarySelector:
    def __init__(self):
        self.points = []
        self.map_points = []
        
        # RealSense setup
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, 30)
        profile = self.pipeline.start(cfg)
        self.depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.intr = profile.get_stream(rs.stream.depth)\
                        .as_video_stream_profile()\
                        .get_intrinsics()
        
        print("Boundary Point Selector")
        print("=======================")
        print("Instructions:")
        print("1. Click 4 points in the camera view to define your boundary")
        print("2. Points will be converted to map coordinates automatically")
        print("3. Press 'r' to reset and start over")
        print("4. Press 'ESC' to quit")
        print("5. The boundary coordinates will be printed for copy-paste")
        print()
        
        # Wait a moment for camera to stabilize
        time.sleep(1)
        
        # Get initial frame for display
        self.current_frame = None
        self.update_frame()
        
        # Setup window and mouse callback
        cv2.namedWindow("Boundary Selector", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Boundary Selector", self.mouse_callback)
        
        self.run()
    
    def update_frame(self):
        """Get the latest frame from camera"""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        cf = aligned.get_color_frame()
        df = aligned.get_depth_frame()
        
        if cf and df:
            self.current_frame = np.asanyarray(cf.get_data())
            self.depth_frame = cv2.medianBlur(np.asanyarray(df.get_data()), 5)
            return True
        return False
    
    def pixel_to_map(self, pixel_x, pixel_y):
        """Convert pixel coordinates to map coordinates"""
        try:
            # Get depth at clicked point
            y0, y1 = max(0, pixel_y - 3), min(FRAME_H, pixel_y + 4)
            x0, x1 = max(0, pixel_x - 3), min(FRAME_W, pixel_x + 4)
            patch = self.depth_frame[y0:y1, x0:x1]
            
            if patch.size == 0:
                return None
                
            z = float(np.median(patch)) * self.depth_scale
            if z <= 0:
                return None
            
            # Convert to camera coordinates
            Xc, Yc, Zc = rs.rs2_deproject_pixel_to_point(self.intr, [pixel_x, pixel_y], z)
            
            # Transform to map coordinates
            P = T_MAP_CAM @ np.array([Xc, Yc, Zc, 1.0], float)
            Xm, Ym, _ = P[:3]
            
            return (float(Xm), float(Ym))
        except Exception as e:
            print(f"Error converting pixel to map: {e}")
            return None
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks"""
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self.points) < 4:
                # Convert to map coordinates
                map_point = self.pixel_to_map(x, y)
                if map_point is not None:
                    self.points.append((x, y))
                    self.map_points.append(map_point)
                    print(f"Point {len(self.points)}: Pixel({x}, {y}) -> Map({map_point[0]:.3f}, {map_point[1]:.3f})")
                    
                    if len(self.points) == 4:
                        self.print_boundary_config()
                else:
                    print(f"Failed to get depth at pixel ({x}, {y}). Try clicking on a visible surface.")
    
    def print_boundary_config(self):
        """Print the boundary configuration for copy-paste"""
        print("\n" + "="*50)
        print("BOUNDARY CONFIGURATION - COPY AND PASTE THIS:")
        print("="*50)
        print("BOUNDARY_POINTS_MAP = np.array([")
        for i, (x, y) in enumerate(self.map_points):
            print(f"    [{x:.3f}, {y:.3f}],  # Point {i+1}")
        print("], dtype=np.float32)")
        print("="*50)
        print("Boundary selection complete! Press 'r' to reset or ESC to quit.")
    
    def draw_boundary(self, img):
        """Draw the current boundary points and lines"""
        # Draw points
        for i, (x, y) in enumerate(self.points):
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(img, f"{i+1}", (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw lines between points
        if len(self.points) > 1:
            for i in range(len(self.points)):
                pt1 = self.points[i]
                pt2 = self.points[(i+1) % len(self.points)]
                if i == len(self.points) - 1 and len(self.points) < 4:
                    continue  # Don't close the polygon until we have 4 points
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)
        
        # Draw instructions
        status = f"Click point {len(self.points)+1}/4" if len(self.points) < 4 else "Complete! Press 'r' to reset"
        cv2.putText(img, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(img, "Press 'r' to reset, ESC to quit", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def reset(self):
        """Reset all points"""
        self.points = []
        self.map_points = []
        print("\nReset! Click 4 new points...")
    
    def run(self):
        """Main loop"""
        while True:
            # Update frame
            if not self.update_frame():
                continue
            
            # Create display image
            display_img = self.current_frame.copy()
            self.draw_boundary(display_img)
            
            # Show image
            cv2.imshow("Boundary Selector", display_img)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('r'):  # Reset
                self.reset()
        
        # Cleanup
        self.pipeline.stop()
        cv2.destroyAllWindows()

def main():
    try:
        selector = BoundarySelector()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()