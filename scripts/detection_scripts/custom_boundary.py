import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO

# === CONFIGURABLE YOLO THRESHOLDS ===
CONF_THRESH = 0.5
IOU_THRESH = 0.4

# === CLASS COLORS ===
TRACK_CLASSES = {
    "teleco": (180, 50, 255),  # pink (BGR)
    "person": (255, 255, 100)   # light blue
}

MODEL_PATH = "/home/commu/Desktop/human_detector_ws/models/best_yolo11m.pt"
FRAME_WIDTH, FRAME_HEIGHT = 640, 480

clicked_points = []
boundary_ready = False
boundary_pts = None

model = YOLO(MODEL_PATH)

# === RealSense ===
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, FRAME_WIDTH, FRAME_HEIGHT, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, FRAME_WIDTH, FRAME_HEIGHT, rs.format.z16, 30)
profile = pipeline.start(config)
align = rs.align(rs.stream.color)

depth_stream = profile.get_stream(rs.stream.depth)
intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

def mouse_callback(event, x, y, flags, param):
    global clicked_points, boundary_ready, boundary_pts
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 4:
        clicked_points.append((x, y))
        if len(clicked_points) == 4:
            boundary_ready = True
            boundary_pts = np.array(clicked_points, dtype=np.int32).reshape((-1, 1, 2))

def get_xyz(depth_frame, x, y):
    x = np.clip(int(x), 0, FRAME_WIDTH - 1)
    y = np.clip(int(y), 0, FRAME_HEIGHT - 1)
    depth = depth_frame.get_distance(x, y)
    point = rs.rs2_deproject_pixel_to_point(intrinsics, [x, y], depth)
    return point

def draw_text_with_bg(img, text, pos, font_scale=0.4, text_color=(255,255,255), bg_color=(0,0,0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    cv2.rectangle(img, (x, y - h - 4), (x + w + 4, y + 4), bg_color, -1)
    cv2.putText(img, text, (x + 2, y), font, font_scale, text_color, thickness)

cv2.namedWindow("Teleco+Human Tracker")
cv2.setMouseCallback("Teleco+Human Tracker", mouse_callback)

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned = align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            continue

        color_image = np.asanyarray(color_frame.get_data())
        display = color_image.copy()

        for pt in clicked_points:
            cv2.circle(display, pt, 5, (0, 255, 0), -1)
        if boundary_ready:
            cv2.polylines(display, [boundary_pts], isClosed=True, color=(0, 0, 255), thickness=2)
        else:
            draw_text_with_bg(display, "Click 4 points to set boundary", (10, 30), font_scale=0.6)

        results = model.predict(source=color_image, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]

        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = model.names[cls_id].lower()
            if class_name not in TRACK_CLASSES:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cx, cy = (x1 + x2) // 2, y2  # base center
            cx = np.clip(cx, 0, FRAME_WIDTH - 1)
            cy = np.clip(cy, 0, FRAME_HEIGHT - 1)

            if boundary_ready:
                inside = cv2.pointPolygonTest(boundary_pts, (float(cx), float(cy)), False)
                if inside < 0:
                    continue

            x, y, z = get_xyz(depth_frame, cx, cy)
            color = TRACK_CLASSES[class_name]

            # Draw bounding box and labels
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            cv2.circle(display, (cx, cy), 3, (255, 255, 255), -1)

            # Draw class label above box
            draw_text_with_bg(display, f"{class_name}", (x1, max(y1 - 10, 10)), font_scale=0.5, text_color=color)

            # Draw XYZ inside bounding box (smaller font)
            xyz_text = f"{x:.2f}, {y:.2f}, {z:.2f}"
            text_x = x1 + 5
            text_y = min(y2 - 5, FRAME_HEIGHT - 10)
            draw_text_with_bg(display, xyz_text, (text_x, text_y), font_scale=0.4)

        cv2.imshow("Teleco+Human Tracker", display)
        key = cv2.waitKey(1)
        if key == 27:
            break
        elif key == ord('r'):
            clicked_points.clear()
            boundary_ready = False
            boundary_pts = None

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
