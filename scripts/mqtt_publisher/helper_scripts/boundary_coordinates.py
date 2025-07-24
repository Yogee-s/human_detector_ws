#!/usr/bin/env python3
import cv2
import numpy as np
import pyrealsense2 as rs
import time

# Configuration
# FRAME_W, FRAME_H = 640, 480
FRAME_W, FRAME_H = 1280, 720
NUM_POINTS = 4

class BoundarySelector:
    def __init__(self):
        self.points = []  # list of (x, y) pixel coords

        # RealSense color-only setup
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)

        print("Click {} points to define a pixel boundary:".format(NUM_POINTS))
        print(" - Left-click to add a point")
        print(" - Press 'r' to reset")
        print(" - Press ESC to finish and print array")

        time.sleep(1)  # allow camera to warm up
        self.current_frame = None
        self.update_frame()

        cv2.namedWindow("Boundary Selector", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Boundary Selector", self.mouse_callback)
        self.run()

    def update_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        cf = aligned.get_color_frame()
        if cf:
            self.current_frame = np.asanyarray(cf.get_data())
            return True
        return False

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < NUM_POINTS:
            self.points.append((x, y))
            print(f"Point {len(self.points)}: ({x}, {y})")
            if len(self.points) == NUM_POINTS:
                print("All points captured. Press ESC to finish.")

    def draw_boundary(self, img):
        # draw selected points
        for idx, (x, y) in enumerate(self.points):
            cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(img, str(idx+1), (x+8, y-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # draw connecting lines
        if len(self.points) > 1:
            for i in range(len(self.points)):
                pt1 = self.points[i]
                pt2 = self.points[(i+1) % len(self.points)]
                if i == len(self.points)-1 and len(self.points) < NUM_POINTS:
                    break
                cv2.line(img, pt1, pt2, (0, 255, 0), 2)

    def print_boundary(self):
        arr = np.array(self.points, dtype=np.int32)
        print("BOUNDARY_POINTS_MAP = np.array([")
        for x, y in arr:
            print(f"    [{x}, {y}],")
        print(f"], dtype=np.int32)")

    def run(self):
        while True:
            if not self.update_frame():
                continue
            disp = self.current_frame.copy()
            self.draw_boundary(disp)
            cv2.imshow("Boundary Selector", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord('r'):
                self.points = []
                print("Points reset. Click again.")
        self.pipeline.stop()
        cv2.destroyAllWindows()
        if len(self.points) == NUM_POINTS:
            self.print_boundary()
        else:
            print("Boundary selection incomplete.")

if __name__ == '__main__':
    BoundarySelector()
