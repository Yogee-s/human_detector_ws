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
from tabulate import tabulate
import time

# ─── USER CONFIG ─────────────────────────────────────────────────────────────
MQTT_USERNAME = "commu"
MQTT_PASSWORD = "zD5%rZ$m/i+W"
MQTT_ADDRESS = "mittsu-talk.jp"
MQTT_PORT = 443
MQTT_PATH = "/mosmos-test2/ws/"
MQTT_TOPIC = "/topic/testing/testroom/ROVER-001/info"
YOLO_MODEL = "/home/commu/Desktop/human_detector_ws/models/best_yolo11m.pt"
FRAME_W, FRAME_H = 640, 480
CONF_THRESH = 0.4
IOU_THRESH = 0.5
TRACKER_CFG = "bytetrack.yaml"

# TRANSFORMATION MATRIX 3 points
# T_MAP_CAM = np.array([
#   [ 0.03230917, -0.53725886,  0.84279834, -3.66181058],
#   [-0.99837390,  0.02227482,  0.05247279,  0.51923760],
#   [-0.04696465, -0.84312321, -0.53566554,  1.60349142],
#   [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
# ], dtype=float)

# TRANSFORMATION MATRIX 4 points
# T_MAP_CAM = np.array([
#   [ 0.03673103, -0.46990984,  0.88194987, -3.88086136],
#   [-0.99835943,  0.02153540,  0.05305343,  0.54558778],
#   [-0.04392348, -0.88245168, -0.46834790,  1.46262758],
#   [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
# ], dtype=float)

# TRANSFORMATION MATRIX 5 points
# T_MAP_CAM = np.array([
#   [ 0.12983621, -0.47795265,  0.86873691, -3.84405157],
#   [-0.98608876,  0.02946711,  0.16358682,  0.17396688],
#   [-0.10378592, -0.87789120, -0.46747784,  1.47143380],
#   [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
# ], dtype=float)


# # TRANSFORMATION MATRIX 6 points
# T_MAP_CAM = np.array([
#   [ 0.11438371, -0.49697971,  0.86019041, -3.77437067],
#   [-0.99078636,  0.00613634,  0.13529499,  0.29111429],
#   [-0.07251728, -0.86774046, -0.49169882,  1.53077601],
#   [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
# ], dtype=float)


# # TRANSFORMATION MATRIX 7 points
# T_MAP_CAM = np.array([
#   [ 0.07631752, -0.50464664,  0.85994616, -3.82267133],
#   [-0.99135697,  0.05389813,  0.11960913,  0.30485382],
#   [-0.10670984, -0.86164190, -0.49617160,  1.53403619],
#   [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
# ], dtype=float)


# TRANSFORMATION MATRIX 8 points
# T_MAP_CAM = np.array([
#   [ 0.38985998, -0.58970809,  0.70728606, -3.21545236],
#   [-0.79475841,  0.17249713,  0.58189674, -1.01032257],
#   [-0.46515403, -0.78897979, -0.40142573,  1.11070835],
#   [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
# ], dtype=float)

# TRANSFORMATION MATRIX 9 points
T_MAP_CAM = np.array([
  [ 0.05886158, -0.51341761,  0.85611779, -3.76887985],
  [-0.99595930,  0.02806597,  0.08530753,  0.44278361],
  [-0.06782617, -0.85767981, -0.50969103,  1.56517224],
  [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
], dtype=float)


# GUI Layout constants
GUI_HEIGHT = 200  # Increased height for data table display
TOTAL_HEIGHT = FRAME_H + GUI_HEIGHT

logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────
collected_data = []  # List of dictionaries containing measurement data
lock = threading.Lock()
latest_map = None  # last robot pose from MQTT
target_id = None  # locked YOLO track ID
available_ids = set()  # set of currently detected IDs
id_history = {}  # dict to track ID consistency: {id: consecutive_frames}
selected_id = None  # ID selected from dropdown
selection_mode = True  # True when selecting ID, False when tracking
dropdown_active = False  # True when dropdown is being shown
dropdown_options = []  # List of available IDs as strings
dropdown_selected_idx = 0  # Currently highlighted option in dropdown

def transform_camera_to_map(cam_point):
    """Transform camera coordinates to map coordinates using the transformation matrix"""
    # Convert to homogeneous coordinates
    cam_homogeneous = np.array([cam_point[0], cam_point[1], cam_point[2], 1.0])
    # Apply transformation
    map_homogeneous = T_MAP_CAM @ cam_homogeneous
    # Return 3D coordinates
    return map_homogeneous[:3]

def print_data_table():
    """Print collected data in a formatted table"""
    if not collected_data:
        print("\nNo data collected yet.")
        return
    
    print("\n" + "="*120)
    print("COLLECTED DATA TABLE")
    print("="*120)
    
    # Prepare table data
    table_data = []
    headers = ["#", "Timestamp", "ID", "Camera X", "Camera Y", "Camera Z", 
               "SLAM X", "SLAM Y", "SLAM Z", "Transformed X", "Transformed Y", "Transformed Z",
               "Error X", "Error Y", "Error Z", "Error Magnitude"]
    
    for i, data in enumerate(collected_data, 1):
        cam = data['camera_coords']
        slam = data['slam_coords']
        trans = data['transformed_coords']
        error = data['error']
        error_mag = data['error_magnitude']
        
        row = [
            i,
            data['timestamp'].strftime("%H:%M:%S"),
            data['target_id'],
            f"{cam[0]:.3f}",
            f"{cam[1]:.3f}",
            f"{cam[2]:.3f}",
            f"{slam[0]:.3f}",
            f"{slam[1]:.3f}",
            f"{slam[2]:.3f}",
            f"{trans[0]:.3f}",
            f"{trans[1]:.3f}",
            f"{trans[2]:.3f}",
            f"{error[0]:.3f}",
            f"{error[1]:.3f}",
            f"{error[2]:.3f}",
            f"{error_mag:.3f}"
        ]
        table_data.append(row)
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    # Print statistics
    if len(collected_data) > 1:
        errors = [data['error_magnitude'] for data in collected_data]
        print(f"\nSTATISTICS:")
        print(f"Total samples: {len(collected_data)}")
        print(f"Average error: {np.mean(errors):.3f}")
        print(f"Standard deviation: {np.std(errors):.3f}")
        print(f"Min error: {np.min(errors):.3f}")
        print(f"Max error: {np.max(errors):.3f}")
    
    print("="*120)

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
    button_y = gui_start_y + 90
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
    
    # Print Table button
    print_x = clear_x + button_width + 10
    print_color = (150, 100, 0)
    cv2.rectangle(img, (print_x, button_y), (print_x + button_width, button_y + button_height),
                 print_color, -1)
    cv2.rectangle(img, (print_x, button_y), (print_x + button_width, button_y + button_height),
                 (200, 200, 200), 2)
    cv2.putText(img, "Print Table", (print_x + 10, button_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_data_summary(img):
    """Draw a summary of collected data points in the GUI area"""
    gui_start_y = FRAME_H
    summary_y = gui_start_y + 130
    
    # Background for data summary
    cv2.rectangle(img, (10, summary_y), (FRAME_W - 10, TOTAL_HEIGHT - 10), (30, 30, 30), -1)
    cv2.rectangle(img, (10, summary_y), (FRAME_W - 10, TOTAL_HEIGHT - 10), (100, 100, 100), 2)
    
    # Title
    cv2.putText(img, "Recent Data Points:", (20, summary_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Show last few data points
    if collected_data:
        recent_data = collected_data[-3:]  # Show last 3 points
        for i, data in enumerate(recent_data):
            y_pos = summary_y + 40 + i * 15
            cam = data['camera_coords']
            slam = data['slam_coords']
            error_mag = data['error_magnitude']
            
            text = f"#{len(collected_data) - len(recent_data) + i + 1}: Cam({cam[0]:.2f},{cam[1]:.2f},{cam[2]:.2f}) SLAM({slam[0]:.2f},{slam[1]:.2f},{slam[2]:.2f}) Err:{error_mag:.3f}"
            cv2.putText(img, text, (20, y_pos),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    else:
        cv2.putText(img, "No data collected yet. Press SPACE to collect.", (20, summary_y + 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1)

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
        dropdown_y = gui_start_y + 45
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
        
        # Check buttons (positioned in GUI area)
        button_y = gui_start_y + 90
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
        
        # Print Table button
        print_x = clear_x + button_width + 10
        if (print_x <= x <= print_x + button_width and
            button_y <= y <= button_y + button_height):
            print_data_table()

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
mqttc = mqtt.Client(client_id="data_collector", transport="websockets")
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

# ─── CLEANUP & SAVE DATA ──────────────────────────────────────────────────────
def cleanup_and_exit(sig, frame):
    print("\n[exit] Shutting down and printing final results...")
    pipeline.stop()
    mqttc.loop_stop()
    cv2.destroyAllWindows()
    
    # Print final data table
    print_data_table()
    
    # Save data to CSV file
    if collected_data:
        import csv
        filename = f"data_collection_{int(time.time())}.csv"
        with open(filename, 'w', newline='') as csvfile:
            fieldnames = ['timestamp', 'target_id', 'camera_x', 'camera_y', 'camera_z',
                         'slam_x', 'slam_y', 'slam_z', 'transformed_x', 'transformed_y', 'transformed_z',
                         'error_x', 'error_y', 'error_z', 'error_magnitude']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for data in collected_data:
                cam = data['camera_coords']
                slam = data['slam_coords']
                trans = data['transformed_coords']
                error = data['error']
                
                writer.writerow({
                    'timestamp': data['timestamp'].strftime("%Y-%m-%d %H:%M:%S"),
                    'target_id': data['target_id'],
                    'camera_x': cam[0], 'camera_y': cam[1], 'camera_z': cam[2],
                    'slam_x': slam[0], 'slam_y': slam[1], 'slam_z': slam[2],
                    'transformed_x': trans[0], 'transformed_y': trans[1], 'transformed_z': trans[2],
                    'error_x': error[0], 'error_y': error[1], 'error_z': error[2],
                    'error_magnitude': data['error_magnitude']
                })
        print(f"\n[saved] Data exported to {filename}")
    
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
cv2.namedWindow("data_collector", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("data_collector", handle_mouse_click)
print("** Data Collection System **")
print(" - Use dropdown below video to select a Teleco ID to track")
print(" - Use 'Auto Select' to choose the most stable ID")
print(" - Hit [SPACE] to collect a data point when tracking")
print(" - Click 'Print Table' or press 'T' to display current data")
print(" - Press Ctrl+C or Esc to finish & export data\n")
print(f" - Using transformation matrix with shape: {T_MAP_CAM.shape}")

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
                
                # ID label with background
                label = f"ID {tid}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(img, (x1, y1-25), (x1+label_size[0]+5, y1-5), color, -1)
                cv2.putText(img, label, (x1+2, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                cv2.putText(img, "TRACKING", (x1, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                break
    
    # Compute current cam_pt
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
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                cx,cy = (x1+x2)//2, (y1+y2)//2
                
                # Use larger patch for more stable depth reading
                patch_size = 5
                patch = depth[max(0,cy-patch_size):cy+patch_size+1,
                             max(0,cx-patch_size):cx+patch_size+1]
                if patch.size:
                    # Use median for stability, but filter out zeros
                    valid_depths = patch[patch > 0]
                    if len(valid_depths) > 0:
                        z = float(np.median(valid_depths)) * depth_scale
                        if z > 0:
                            cam_pt = rs.rs2_deproject_pixel_to_point(intrinsics, [cx,cy], z)
                break
    
    with lock:
        rob_pt = latest_map.copy() if latest_map is not None else None
    
    # Draw GUI area background
    draw_gui_area(img)
    
    # Draw GUI elements in the dedicated area below video
    gui_start_y = FRAME_H
    
    # Status text positioned at the top of GUI area
    status_y = gui_start_y + 20
    if selection_mode:
        status_text = "MODE: Selection - Choose an ID to track"
        status_color = (0, 255, 255)
    else:
        status_text = f"MODE: Tracking ID {target_id} - Press SPACE to collect data"
        status_color = (0, 255, 0)
    cv2.putText(img, status_text, (10, status_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Dropdown positioned below status text
    dropdown_y = gui_start_y + 45
    draw_dropdown(img, 10, dropdown_y, 200, 30, dropdown_options, dropdown_selected_idx, dropdown_active)
    
    # Buttons positioned below dropdown
    draw_buttons(img, not selection_mode)
    
    # Data summary
    draw_data_summary(img)
    
    # Data overlay: positioned in video area (top-left)
    y = 30
    x0 = 10
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.6, 2
    bg = (50,50,50)
    fg = (255,255,255)
    spacing = 12
    
    # Camera coordinates
    if cam_pt is None:
        txt1 = "Cam: Waiting for detection..."
    else:
        txt1 = f"Cam: ({cam_pt[0]:.3f}, {cam_pt[1]:.3f}, {cam_pt[2]:.3f})"
    
    # Draw text with background
    text_size = cv2.getTextSize(txt1, font, fs, th)[0]
    cv2.rectangle(img, (x0-5, y-text_size[1]-5), (x0+text_size[0]+5, y+5), bg, -1)
    cv2.putText(img, txt1, (x0, y), font, fs, fg, th)
    y += text_size[1] + spacing
    
    # SLAM coordinates
    if rob_pt is None:
        txt2 = "SLAM: No robot position"
    else:
        txt2 = f"SLAM: ({rob_pt[0]:.3f}, {rob_pt[1]:.3f}, {rob_pt[2]:.3f})"
    
    text_size = cv2.getTextSize(txt2, font, fs, th)[0]
    cv2.rectangle(img, (x0-5, y-text_size[1]-5), (x0+text_size[0]+5, y+5), bg, -1)
    cv2.putText(img, txt2, (x0, y), font, fs, fg, th)
    y += text_size[1] + spacing
    
    # Transformation status
    if cam_pt is not None:
        transformed_pt = transform_camera_to_map(cam_pt)
        txt3 = f"Trans: ({transformed_pt[0]:.3f}, {transformed_pt[1]:.3f}, {transformed_pt[2]:.3f})"
    else:
        txt3 = "Trans: Waiting for camera data..."
    
    text_size = cv2.getTextSize(txt3, font, fs, th)[0]
    cv2.rectangle(img, (x0-5, y-text_size[1]-5), (x0+text_size[0]+5, y+5), bg, -1)
    cv2.putText(img, txt3, (x0, y), font, fs, fg, th)
    y += text_size[1] + spacing
    
    # Error calculation if both points available
    if cam_pt is not None and rob_pt is not None:
        transformed_pt = transform_camera_to_map(cam_pt)
        error = transformed_pt - rob_pt
        error_magnitude = np.linalg.norm(error)
        txt4 = f"Error: {error_magnitude:.3f}m"
    else:
        txt4 = "Error: Need both positions"
    
    text_size = cv2.getTextSize(txt4, font, fs, th)[0]
    cv2.rectangle(img, (x0-5, y-text_size[1]-5), (x0+text_size[0]+5, y+5), bg, -1)
    cv2.putText(img, txt4, (x0, y), font, fs, fg, th)
    y += text_size[1] + spacing
    
    # Data collection count
    txt5 = f"Collected: {len(collected_data)} points"
    text_size = cv2.getTextSize(txt5, font, fs, th)[0]
    cv2.rectangle(img, (x0-5, y-text_size[1]-5), (x0+text_size[0]+5, y+5), bg, -1)
    cv2.putText(img, txt5, (x0, y), font, fs, fg, th)
    
    # Show the image
    cv2.imshow("data_collector", img)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == 27:  # ESC key
        break
    elif key == ord(' '):  # SPACE key - collect data point
        if not selection_mode and target_id is not None and cam_pt is not None and rob_pt is not None:
            # Calculate transformed coordinates and error
            transformed_pt = transform_camera_to_map(cam_pt)
            error = transformed_pt - rob_pt
            error_magnitude = np.linalg.norm(error)
            
            # Create data point
            data_point = {
                'timestamp': time.time(),
                'target_id': target_id,
                'camera_coords': np.array(cam_pt),
                'slam_coords': np.array(rob_pt),
                'transformed_coords': transformed_pt,
                'error': error,
                'error_magnitude': error_magnitude
            }
            
            # Convert timestamp to datetime for display
            import datetime
            data_point['timestamp'] = datetime.datetime.fromtimestamp(data_point['timestamp'])
            
            with lock:
                collected_data.append(data_point)
            
            print(f"[data] Collected point #{len(collected_data)}: "
                  f"Cam({cam_pt[0]:.3f},{cam_pt[1]:.3f},{cam_pt[2]:.3f}) "
                  f"SLAM({rob_pt[0]:.3f},{rob_pt[1]:.3f},{rob_pt[2]:.3f}) "
                  f"Error: {error_magnitude:.3f}m")
        else:
            if selection_mode:
                print("[data] Cannot collect: Select an ID first")
            elif target_id is None:
                print("[data] Cannot collect: No target ID")
            elif cam_pt is None:
                print("[data] Cannot collect: No camera detection")
            elif rob_pt is None:
                print("[data] Cannot collect: No SLAM position")
    
    elif key == ord('t') or key == ord('T'):  # T key - print table
        print_data_table()
    
    elif key == ord('c') or key == ord('C'):  # C key - clear selection
        clear_selection()
    
    elif key == ord('a') or key == ord('A'):  # A key - auto select
        if selection_mode:
            auto_select_id()
    
    # Handle dropdown navigation with arrow keys
    elif key == 81 or key == 82:  # Up/Down arrow keys
        if dropdown_active and dropdown_options:
            if key == 81:  # Up arrow
                dropdown_selected_idx = (dropdown_selected_idx - 1) % len(dropdown_options)
            else:  # Down arrow
                dropdown_selected_idx = (dropdown_selected_idx + 1) % len(dropdown_options)
    
    elif key == 13:  # Enter key - select from dropdown
        if dropdown_active and dropdown_options and dropdown_selected_idx < len(dropdown_options):
            selected_id = int(dropdown_options[dropdown_selected_idx])
            target_id = selected_id
            selection_mode = False
            dropdown_active = False
            print(f"[gui] Selected ID {selected_id} for tracking")
    
    frame_count += 1

# Cleanup on normal exit
cleanup_and_exit(None, None)