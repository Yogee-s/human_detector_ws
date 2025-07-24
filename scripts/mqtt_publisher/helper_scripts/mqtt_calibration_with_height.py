# ─── KABSCH SOLVER WITH 3D HEIGHT MAPPING ──────────────────────────────────
#!/usr/bin/env python3
import signal
import sys
import threading
import json
import logging
import numpy as np
import paho.mqtt.client as mqtt
import pyrealsense2 as rs
import cv2
from ultralytics import YOLO
# ─── USER CONFIG ─────────────────────────────────────────────────────────────
MQTT_USERNAME = "commu"
MQTT_PASSWORD = "zD5%rZ$m/i+W"
MQTT_ADDRESS = "mittsu-talk.jp"
MQTT_PORT = 443
MQTT_PATH = "/mosmos-test2/ws/"
MQTT_TOPIC = "/topic/testing/testroom/ROVER-001/info"
YOLO_MODEL = "/home/commu/Desktop/human_detector_ws/models/best_yolo11m.pt"
# FRAME_W, FRAME_H = 640, 480
FRAME_W, FRAME_H = 1280, 720
CONF_THRESH = 0.4
IOU_THRESH = 0.5
TRACKER_CFG = "bytetrack.yaml"

# Height configuration
ROBOT_HEIGHT_IN_MAP = 1.36  # Known height of robot in SLAM map coordinates (meters)

# GUI Layout constants
GUI_HEIGHT = 150  # Height for GUI area below video
TOTAL_HEIGHT = FRAME_H + GUI_HEIGHT
# ───────────────────────────────────────────────────────────────────────────────
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────
cam_pts = [] # list of np.array([Xc,Yc,Zc]) - now includes height from top of bbox
map_pts = [] # list of np.array([x_map,y_map,z_map]) - now includes height in map coords
lock = threading.Lock()
latest_map = None # last robot pose from MQTT
target_id = None # locked YOLO track ID
available_ids = set() # set of currently detected IDs
id_history = {} # dict to track ID consistency: {id: consecutive_frames}
selected_id = None # ID selected from dropdown
selection_mode = True # True when selecting ID, False when tracking
dropdown_active = False # True when dropdown is being shown
dropdown_options = [] # List of available IDs as strings
dropdown_selected_idx = 0 # Currently highlighted option in dropdown

# ─── GUI HELPER FUNCTIONS ────────────────────────────────────────────────────
def draw_dropdown(img, x, y, width, height, options, selected_idx, active):
    """Draw a dropdown menu on the OpenCV image"""
    # Main dropdown box
    bg_color = (40, 40, 40) if active else (60, 60, 60)
    border_color = (0, 255, 0) if active else (100, 100, 100)
    cv2.rectangle(img, (x, y), (x + width, y + height), bg_color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), border_color, 2)
    
    # Current selection text
    if options and selected_idx < len(options):
        text = f"ID: {options[selected_idx]}"
    else:
        text = "Select ID"
    cv2.putText(img, text, (x + 10, y + height//2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Dropdown arrow
    arrow_x = x + width - 20
    arrow_y = y + height//2
    cv2.arrowedLine(img, (arrow_x, arrow_y - 5), (arrow_x, arrow_y + 5),
                    (255, 255, 255), 2)
    
    # If active, show dropdown options (expand upward to stay in GUI area)
    if active and options:
        option_height = 30
        dropdown_height = len(options) * option_height
        # Background for options - expand upward from dropdown
        options_y = y - dropdown_height
        cv2.rectangle(img, (x, options_y), (x + width, y),
                     (30, 30, 30), -1)
        cv2.rectangle(img, (x, options_y), (x + width, y),
                     (100, 100, 100), 2)
        
        # Draw each option
        for i, option in enumerate(options):
            option_y = options_y + i * option_height
            # Highlight selected option
            if i == selected_idx:
                cv2.rectangle(img, (x + 2, option_y + 2),
                             (x + width - 2, option_y + option_height - 2),
                             (0, 100, 200), -1)
            cv2.putText(img, f"ID {option}", (x + 10, option_y + option_height//2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_buttons(img, tracking_active):
    """Draw control buttons on the image"""
    # Position buttons in the GUI area below dropdown
    gui_start_y = FRAME_H
    button_y = gui_start_y + 90  # Changed from +80 to +90 to account for dropdown repositioning
    button_height = 30
    button_width = 120
    
    # Auto Select button
    auto_x = 10
    auto_color = (0, 150, 0) if not tracking_active else (100, 100, 100)
    cv2.rectangle(img, (auto_x, button_y), (auto_x + button_width, button_y + button_height),
                 auto_color, -1)
    cv2.rectangle(img, (auto_x, button_y), (auto_x + button_width, button_y + button_height),
                 (200, 200, 200), 2)
    cv2.putText(img, "Auto Select", (auto_x + 10, button_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Clear/Stop button
    clear_x = auto_x + button_width + 10
    clear_text = "Stop Track" if tracking_active else "Clear"
    clear_color = (0, 0, 150) if tracking_active else (150, 0, 0)
    cv2.rectangle(img, (clear_x, button_y), (clear_x + button_width, button_y + button_height),
                 clear_color, -1)
    cv2.rectangle(img, (clear_x, button_y), (clear_x + button_width, button_y + button_height),
                 (200, 200, 200), 2)
    cv2.putText(img, clear_text, (clear_x + 10, button_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_gui_area(img):
    """Draw the GUI background area below the video"""
    gui_start_y = FRAME_H
    # Draw dark background for GUI area
    cv2.rectangle(img, (0, gui_start_y), (FRAME_W, TOTAL_HEIGHT), (40, 40, 40), -1)
    # Draw separator line
    cv2.line(img, (0, gui_start_y), (FRAME_W, gui_start_y), (100, 100, 100), 2)

def handle_mouse_click(event, x, y, flags, param):
    """Handle mouse clicks on the OpenCV window"""
    global dropdown_active, dropdown_selected_idx, selected_id, target_id, selection_mode
    
    if event == cv2.EVENT_LBUTTONDOWN:
        gui_start_y = FRAME_H
        
        # Check dropdown area (positioned in GUI area)
        dropdown_x = 10
        dropdown_y = gui_start_y + 45  # Updated position
        dropdown_width, dropdown_height = 200, 30
        
        if (dropdown_x <= x <= dropdown_x + dropdown_width and
            dropdown_y <= y <= dropdown_y + dropdown_height):
            if selection_mode and dropdown_options:
                dropdown_active = not dropdown_active
        
        # Check dropdown options if active (expand upward in GUI area)
        elif dropdown_active and dropdown_options:
            option_height = 30
            dropdown_height_total = len(dropdown_options) * option_height
            options_start_y = dropdown_y - dropdown_height_total
            
            for i, option in enumerate(dropdown_options):
                option_y = options_start_y + i * option_height
                if (dropdown_x <= x <= dropdown_x + dropdown_width and
                    option_y <= y <= option_y + option_height):
                    dropdown_selected_idx = i
                    selected_id = int(dropdown_options[i])
                    target_id = selected_id
                    selection_mode = False
                    dropdown_active = False
                    print(f"[gui] Selected ID {selected_id} for tracking")
                    break
        
        # Check buttons (positioned in GUI area with updated coordinates)
        button_y = gui_start_y + 90  # Updated from +80 to +90
        button_height = 30
        button_width = 120
        
        # Auto Select button
        auto_x = 10
        if (auto_x <= x <= auto_x + button_width and
            button_y <= y <= button_y + button_height and selection_mode):
            auto_select_id()
        
        # Clear/Stop button
        clear_x = auto_x + button_width + 10
        if (clear_x <= x <= clear_x + button_width and
            button_y <= y <= button_y + button_height):
            clear_selection()

def auto_select_id():
    """Auto-select the most consistently detected ID"""
    global id_history, selected_id, target_id, selection_mode, dropdown_selected_idx
    if id_history:
        best_id = max(id_history.keys(), key=lambda k: id_history[k])
        selected_id = best_id
        target_id = best_id
        selection_mode = False
        # Update dropdown selection
        if str(best_id) in dropdown_options:
            dropdown_selected_idx = dropdown_options.index(str(best_id))
        print(f"[gui] Auto-selected ID {best_id} for tracking")

def clear_selection():
    """Clear the current selection and return to selection mode"""
    global selected_id, target_id, selection_mode, dropdown_active
    selected_id = None
    target_id = None
    selection_mode = True
    dropdown_active = False
    print("[gui] Cleared ID selection - returning to selection mode")

def get_3d_point_from_bbox(box, depth, intrinsics, depth_scale, use_top=False):
    """
    Extract 3D point from bounding box.
    If use_top=True, uses top-middle of bbox for height measurement.
    If use_top=False, uses center of bbox (original behavior).
    """
    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
    
    if use_top:
        # Use top-middle of bounding box for height measurement
        cx = (x1 + x2) // 2
        cy = y1  # Top of the bounding box
    else:
        # Use center of bounding box (original behavior)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
    
    # Use larger patch for more stable depth reading
    patch_size = 5
    patch = depth[max(0, cy - patch_size):cy + patch_size + 1,
                 max(0, cx - patch_size):cx + patch_size + 1]
    
    if patch.size:
        # Use median for stability, but filter out zeros
        valid_depths = patch[patch > 0]
        if len(valid_depths) > 0:
            z = float(np.median(valid_depths)) * depth_scale
            if z > 0:
                return np.array(rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], z), dtype=float)
    
    return None

def compute_rigid_transform(cam_pts: np.ndarray, map_pts: np.ndarray) -> np.ndarray:
    """Compute 3D to 3D rigid transformation using Kabsch algorithm"""
    assert cam_pts.shape == map_pts.shape and cam_pts.shape[0] >= 3
    assert cam_pts.shape[1] == 3 and map_pts.shape[1] == 3, "Both point sets must be 3D"
    
    # Compute centroids
    c_cam = cam_pts.mean(axis=0)
    c_map = map_pts.mean(axis=0)
    
    # Center the point sets
    A = cam_pts - c_cam
    B = map_pts - c_map
    
    # Compute the cross-covariance matrix
    H = A.T @ B
    
    # SVD decomposition
    U, S, Vt = np.linalg.svd(H)
    
    # Compute rotation matrix
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = c_map - R @ c_cam
    
    # Build 4x4 transformation matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    
    return T

# ─── MQTT CALLBACKS ──────────────────────────────────────────────────────────
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        client.subscribe(MQTT_TOPIC)
        print(f"[mqtt] connected, subscribed to {MQTT_TOPIC}")
    else:
        print(f"[mqtt] failed to connect: rc={rc}")

def on_message(client, userdata, msg):
    global latest_map
    raw = msg.payload.decode('utf-8', errors='ignore')
    try:
        d = json.loads(raw)
        tele = d.get("teleco") or {}
        if "x" in tele and "y" in tele and "z" in tele:
            x,y,z = float(tele["x"]), float(tele["y"]), float(tele["z"])
        elif d.get("pose",{}).get("position"):
            p = d["pose"]["position"]
            x,y,z = float(p["x"]), float(p["y"]), float(p["z"])
        else:
            return
        with lock:
            latest_map = np.array([x,y,z], dtype=float)
    except:
        pass

# ─── SETUP MQTT ──────────────────────────────────────────────────────────────
mqttc = mqtt.Client(client_id="calib", transport="websockets")
mqttc.tls_set()
mqttc.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
mqttc.ws_set_options(path=MQTT_PATH)
mqttc.on_connect = on_connect
mqttc.on_message = on_message
mqttc.connect(MQTT_ADDRESS, MQTT_PORT)
mqttc.loop_start()

# ─── REALSENSE + YOLO SETUP ──────────────────────────────────────────────────
pipeline = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, FRAME_W, FRAME_H, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, FRAME_W, FRAME_H, rs.format.z16, 30)
profile = pipeline.start(cfg)
align = rs.align(rs.stream.color)
intrinsics = profile.get_stream(rs.stream.depth) \
             .as_video_stream_profile() \
             .get_intrinsics()
depth_scale = profile.get_device() \
              .first_depth_sensor() \
              .get_depth_scale()
yolo = YOLO(YOLO_MODEL)

# ─── CLEANUP & PRINT TRANSFORM ────────────────────────────────────────────────
def cleanup_and_exit(sig, frame):
    print("\n[exit] computing final 3D to 3D transform…")
    pipeline.stop()
    mqttc.loop_stop()
    cv2.destroyAllWindows()
    if len(cam_pts) >= 3:
        C = np.stack(cam_pts, axis=0)
        M = np.stack(map_pts, axis=0)
        T = compute_rigid_transform(C, M)
        print("\n# 3D to 3D Transformation Matrix (Camera -> SLAM Map)")
        print("# This transforms camera 3D coordinates (including height) to SLAM map 3D coordinates")
        print("T_MAP_CAM = np.array([")
        for row in T:
            print("  [{: .8f}, {: .8f}, {: .8f}, {: .8f}],".format(*row))
        print("], dtype=float)")
        print(f"\n# Calibrated with {len(cam_pts)} point pairs")
        print("# Usage: map_point_3d = T_MAP_CAM @ np.append(cam_point_3d, 1)[:3]")
    else:
        print(f"need ≥3 points, got {len(cam_pts)}")
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
cv2.namedWindow("calib", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("calib", handle_mouse_click)
print("** 3D Height Calibration running **")
print(" - Use dropdown below video to select a Teleco ID to track")
print(" - Use 'Auto Select' to choose the most stable ID")
print(" - Hit [SPACE] to record a 3D pair when tracking")
print(" - Height is measured from top of bounding box")
print(f" - Robot height in map coordinates: {ROBOT_HEIGHT_IN_MAP}m")
print(" - Press Ctrl+C or Esc to finish & compute the 3D transform\n")

frame_count = 0
while True:
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    cf = aligned.get_color_frame()
    df = aligned.get_depth_frame()
    if not cf or not df:
        continue
    
    # Get camera image
    camera_img = np.asanyarray(cf.get_data())
    depth = np.asanyarray(df.get_data())
    
    # Create extended image with GUI area
    img = np.zeros((TOTAL_HEIGHT, FRAME_W, 3), dtype=np.uint8)
    img[:FRAME_H, :FRAME_W] = camera_img  # Place camera feed at top
    
    # YOLO detection - only run if in selection mode OR tracking specific ID
    current_frame_ids = set()
    if selection_mode:
        # Run full detection to find available IDs
        res = yolo.track(
            camera_img,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            tracker=TRACKER_CFG,
            persist=True
        )[0]
        
        # Track available IDs and update history for stability
        for box in res.boxes:
            cls = int(box.cls[0])
            if yolo.names.get(cls,"").lower() != "teleco":
                continue
            rid = box.id
            if rid is None:
                continue
            tid = int(rid[0])
            current_frame_ids.add(tid)
            
            # Update ID history for stability tracking
            if tid in id_history:
                id_history[tid] += 1
            else:
                id_history[tid] = 1
        
        # Decay history for IDs not seen in current frame
        for tid in list(id_history.keys()):
            if tid not in current_frame_ids:
                id_history[tid] = max(0, id_history[tid] - 2)
                if id_history[tid] == 0:
                    del id_history[tid]
        
        # Update dropdown options
        if current_frame_ids != set(map(int, dropdown_options)):
            dropdown_options = [str(tid) for tid in sorted(current_frame_ids)]
            if dropdown_options and dropdown_selected_idx >= len(dropdown_options):
                dropdown_selected_idx = 0
        
        # Draw all detected telecos in selection mode
        for box in res.boxes:
            cls = int(box.cls[0])
            if yolo.names.get(cls,"").lower() != "teleco":
                continue
            rid = box.id
            if rid is None:
                continue
            tid = int(rid[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            
            # Blue for available IDs in selection mode
            color = (255,100,0)
            thickness = 2
            cv2.rectangle(img, (x1,y1),(x2,y2), color, thickness)
            
            # Draw line at top-middle to show height measurement point
            cx = (x1 + x2) // 2
            cv2.line(img, (cx - 10, y1), (cx + 10, y1), (0, 255, 255), 2)
            cv2.circle(img, (cx, y1), 3, (0, 255, 255), -1)
            
            # ID label with background
            label = f"ID {tid}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img, (x1, y1-25), (x1+label_size[0]+5, y1-5), color, -1)
            cv2.putText(img, label, (x1+2, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(img, "Available", (x1, y2+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
    else:
        # Tracking mode - only detect the selected ID
        res = yolo.track(
            camera_img,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            tracker=TRACKER_CFG,
            persist=True
        )[0]
        
        # Only draw the tracked robot
        for box in res.boxes:
            cls = int(box.cls[0])
            if yolo.names.get(cls,"").lower() != "teleco":
                continue
            rid = box.id
            if rid is None:
                continue
            tid = int(rid[0])
            
            # Only show the selected ID
            if tid == target_id:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                
                # Green for tracked robot
                color = (0,255,0)
                thickness = 3
                cv2.rectangle(img, (x1,y1),(x2,y2), color, thickness)
                
                # Draw line at top-middle to show height measurement point
                cx = (x1 + x2) // 2
                cv2.line(img, (cx - 15, y1), (cx + 15, y1), (0, 255, 255), 3)
                cv2.circle(img, (cx, y1), 4, (0, 255, 255), -1)
                cv2.putText(img, "HEIGHT", (cx - 30, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                # ID label with background
                label = f"ID {tid}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(img, (x1, y1-25), (x1+label_size[0]+5, y1-5), color, -1)
                cv2.putText(img, label, (x1+2, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(img, "TRACKING", (x1, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                break
    
    # Compute current cam_pt (using top of bounding box for height)
    cam_pt = None
    if target_id is not None and not selection_mode:
        for box in res.boxes:
            cls = int(box.cls[0])
            if yolo.names.get(cls,"").lower() != "teleco":
                continue
            rid = box.id
            if rid is None:
                continue
            tid = int(rid[0])
            if tid == target_id:
                cam_pt = get_3d_point_from_bbox(box, depth, intrinsics, depth_scale, use_top=True)
                break
    
    with lock:
        rob_pt = latest_map.copy() if latest_map is not None else None
    
    # Draw GUI area background
    draw_gui_area(img)
    
    # Draw GUI elements in the dedicated area below video with proper spacing
    gui_start_y = FRAME_H
    
    # Status text positioned at the top of GUI area
    status_y = gui_start_y + 20
    if selection_mode:
        status_text = "MODE: Selection - Choose an ID to track"
        status_color = (0, 255, 255)
    else:
        status_text = f"MODE: Tracking ID {target_id} (3D Height)"
        status_color = (0, 255, 0)
    cv2.putText(img, status_text, (10, status_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Dropdown positioned below status text with proper spacing
    dropdown_y = gui_start_y + 45  # Moved down from +30 to +45
    draw_dropdown(img, 10, dropdown_y, 200, 30, dropdown_options, dropdown_selected_idx, dropdown_active)
    
    # Buttons positioned below dropdown with proper spacing
    draw_buttons(img, not selection_mode)
    
    # Data overlay: positioned in video area (top-left)
    y = 30
    x0 = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.6, 2
    bg = (50,50,50)
    fg = (255,255,255)
    spacing = 12
    
    # Camera coordinates (3D with height from top of bbox)
    if cam_pt is None:
        txt1 = "Cam: Waiting for detection..."
    else:
        txt1 = f"Cam: X:{cam_pt[0]:.2f} Y:{cam_pt[1]:.2f} Z:{cam_pt[2]:.2f} (top)"
    (w1,h1),_ = cv2.getTextSize(txt1, font, fs, th)
    cv2.rectangle(img, (x0-4, y-h1-4), (x0+w1+4, y+4), bg, -1)
    cv2.putText(img, txt1, (x0, y), font, fs, fg, th)
    
    # Robot coordinates (now 3D with height)
    y += h1 + spacing
    if rob_pt is None:
        txt2 = "Rob: Waiting for MQTT data..."
    else:
        # For calibration, we assume the robot height is known and constant
        rob_3d = np.array([rob_pt[0], rob_pt[1], ROBOT_HEIGHT_IN_MAP])
        txt2 = f"Rob: X:{rob_3d[0]:.2f} Y:{rob_3d[1]:.2f} Z:{rob_3d[2]:.2f} (fixed)"
    (w2,h2),_ = cv2.getTextSize(txt2, font, fs, th)
    cv2.rectangle(img, (x0-4, y-h2-4), (x0+w2+4, y+4), bg, -1)
    cv2.putText(img, txt2, (x0, y), font, fs, fg, th)
    
    # Pairs recorded
    y += h2 + spacing
    txt3 = f"3D Pairs recorded: {len(cam_pts)}"
    (w3,h3),_ = cv2.getTextSize(txt3, font, fs, th)
    cv2.rectangle(img, (x0-4, y-h3-4), (x0+w3+4, y+4), bg, -1)
    cv2.putText(img, txt3, (x0, y), font, fs, fg, th)
    
    cv2.imshow("calib", img)
    
    key = cv2.waitKey(1) & 0xFF
    
    # SPACE = record one 3D pair (only works when tracking)
    if key == ord(' '):
        if selection_mode or target_id is None:
            print("[warning] Please select a Teleco ID first")
        else:
            print(f"[input] SPACE pressed - attempting to record 3D pair for ID {target_id}")
            picked = None
            for box in res.boxes:
                cls = int(box.cls[0])
                if yolo.names.get(cls,"").lower() != "teleco":
                    continue
                rid = box.id
                if rid is None:
                    continue
                tid = int(rid[0])
                if tid == target_id:
                    cam_3d = get_3d_point_from_bbox(box, depth, intrinsics, depth_scale, use_top=True)
                    if cam_3d is not None:
                        picked = (tid, cam_3d)
                    break
            
            with lock:
                if picked and latest_map is not None:
                    tid, cam_3d = picked
                    # Create 3D map point with known robot height
                    map_3d = np.array([latest_map[0], latest_map[1], ROBOT_HEIGHT_IN_MAP], dtype=float)
                    cam_pts.append(cam_3d)
                    map_pts.append(map_3d)
                    print(f"[recorded] 3D pair #{len(cam_pts)}: cam {cam_3d.round(3)} ↔ map {map_3d.round(3)}")
                else:
                    print("[warning] skipped—no detection or no robot pose")
    
    # ESC = exit + compute transform
    if key == 27:
        cleanup_and_exit(None, None)
    
    frame_count += 1