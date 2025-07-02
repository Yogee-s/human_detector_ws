#!/usr/bin/env python3
import signal
import sys
import threading
import json
import logging
import time
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
FRAME_W, FRAME_H = 640, 480
CONF_THRESH = 0.4
IOU_THRESH = 0.5
TRACKER_CFG = "bytetrack.yaml"

# ─── CALIBRATION CONFIG ──────────────────────────────────────────────────────
SAMPLE_INTERVAL = 3.0  # seconds between automatic samples
MIN_MOVEMENT_THRESHOLD = 0.2  # minimum distance moved to accept new sample
MAX_SAMPLES = 50  # maximum number of samples to keep
OUTLIER_THRESHOLD = 2.0  # standard deviations for outlier detection
MIN_SAMPLES_FOR_OUTLIER_CHECK = 10  # minimum samples before outlier detection

# GUI Layout constants
GUI_HEIGHT = 120
TOTAL_HEIGHT = FRAME_H + GUI_HEIGHT

# ───────────────────────────────────────────────────────────────────────────────
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────
cam_pts = []  # list of np.array([Xc,Yc,Zc])
map_pts = []  # list of np.array([x_map,y_map,z_map])
lock = threading.Lock()
latest_map = None
target_id = None
available_ids = set()
id_history = {}
selected_id = None
selection_mode = True
dropdown_active = False
dropdown_options = []
dropdown_selected_idx = 0

# ─── CALIBRATION STATE ────────────────────────────────────────────────────────
calibration_active = False
last_sample_time = 0
last_sample_position = None
current_transform = None

def is_outlier(new_cam_pt, new_map_pt):
    """Check if new point pair is an outlier based on existing data"""
    if len(cam_pts) < MIN_SAMPLES_FOR_OUTLIER_CHECK:
        return False
    
    # Calculate distances from existing points
    cam_distances = [np.linalg.norm(new_cam_pt - cp) for cp in cam_pts]
    map_distances = [np.linalg.norm(new_map_pt - mp) for mp in map_pts]
    
    # Check if distances are reasonable (not too far from existing points)
    cam_mean, cam_std = np.mean(cam_distances), np.std(cam_distances)
    map_mean, map_std = np.mean(map_distances), np.std(map_distances)
    
    cam_outlier = abs(cam_distances[-1] - cam_mean) > OUTLIER_THRESHOLD * cam_std if cam_std > 0 else False
    map_outlier = abs(map_distances[-1] - map_mean) > OUTLIER_THRESHOLD * map_std if map_std > 0 else False
    
    return cam_outlier or map_outlier

def has_moved_enough(new_map_pt):
    """Check if robot has moved enough since last sample"""
    global last_sample_position
    if last_sample_position is None:
        return True
    
    distance = np.linalg.norm(new_map_pt - last_sample_position)
    return distance >= MIN_MOVEMENT_THRESHOLD

def add_sample(cam_pt, map_pt):
    """Add a new sample point with outlier detection"""
    global last_sample_position, current_transform
    
    # Check for outliers
    if is_outlier(cam_pt, map_pt):
        print(f"[outlier] Rejected sample - cam {cam_pt.round(3)} ↔ map {map_pt.round(3)}")
        return False
    
    # Add the sample
    cam_pts.append(cam_pt.copy())
    map_pts.append(map_pt.copy())
    last_sample_position = map_pt.copy()
    
    # Keep only the most recent samples
    if len(cam_pts) > MAX_SAMPLES:
        cam_pts.pop(0)
        map_pts.pop(0)
    
    print(f"[sample] #{len(cam_pts)}: cam {cam_pt.round(3)} ↔ map {map_pt.round(3)}")
    
    # Update transform if we have enough points
    if len(cam_pts) >= 3:
        try:
            C = np.stack(cam_pts, axis=0)
            M = np.stack(map_pts, axis=0)
            current_transform = compute_rigid_transform(C, M)
        except Exception as e:
            print(f"[error] Failed to compute transform: {e}")
            current_transform = None
    
    return True

# ─── GUI FUNCTIONS ────────────────────────────────────────────────────────────
def draw_dropdown(img, x, y, width, height, options, selected_idx, active):
    """Draw a dropdown menu on the OpenCV image"""
    bg_color = (40, 40, 40) if active else (60, 60, 60)
    border_color = (0, 255, 0) if active else (100, 100, 100)
    cv2.rectangle(img, (x, y), (x + width, y + height), bg_color, -1)
    cv2.rectangle(img, (x, y), (x + width, y + height), border_color, 2)
    
    if options and selected_idx < len(options):
        text = f"ID: {options[selected_idx]}"
    else:
        text = "Select ID"
    cv2.putText(img, text, (x + 10, y + height//2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    arrow_x = x + width - 20
    arrow_y = y + height//2
    cv2.arrowedLine(img, (arrow_x, arrow_y - 5), (arrow_x, arrow_y + 5),
                    (255, 255, 255), 2)
    
    if active and options:
        option_height = 30
        dropdown_height = len(options) * option_height
        options_y = y - dropdown_height
        cv2.rectangle(img, (x, options_y), (x + width, y), (30, 30, 30), -1)
        cv2.rectangle(img, (x, options_y), (x + width, y), (100, 100, 100), 2)
        
        for i, option in enumerate(options):
            option_y = options_y + i * option_height
            if i == selected_idx:
                cv2.rectangle(img, (x + 2, option_y + 2),
                             (x + width - 2, option_y + option_height - 2),
                             (0, 100, 200), -1)
            cv2.putText(img, f"ID {option}", (x + 10, option_y + option_height//2 + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_calibration_button(img):
    """Draw start/stop calibration button"""
    gui_start_y = FRAME_H
    button_y = gui_start_y + 80
    button_height = 30
    button_width = 150
    button_x = 250
    
    if calibration_active:
        color = (0, 0, 150)
        text = "Stop Calibration"
    else:
        color = (0, 150, 0)
        text = "Start Calibration"
    
    cv2.rectangle(img, (button_x, button_y), (button_x + button_width, button_y + button_height),
                 color, -1)
    cv2.rectangle(img, (button_x, button_y), (button_x + button_width, button_y + button_height),
                 (200, 200, 200), 2)
    cv2.putText(img, text, (button_x + 10, button_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_auto_select_button(img):
    """Draw auto select button"""
    gui_start_y = FRAME_H
    button_y = gui_start_y + 80
    button_height = 30
    button_width = 120
    auto_x = 10
    
    auto_color = (0, 150, 0) if selection_mode else (100, 100, 100)
    cv2.rectangle(img, (auto_x, button_y), (auto_x + button_width, button_y + button_height),
                 auto_color, -1)
    cv2.rectangle(img, (auto_x, button_y), (auto_x + button_width, button_y + button_height),
                 (200, 200, 200), 2)
    cv2.putText(img, "Auto Select", (auto_x + 10, button_y + 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

def draw_gui_area(img):
    """Draw the GUI background area below the video"""
    gui_start_y = FRAME_H
    cv2.rectangle(img, (0, gui_start_y), (FRAME_W, TOTAL_HEIGHT), (40, 40, 40), -1)
    cv2.line(img, (0, gui_start_y), (FRAME_W, gui_start_y), (100, 100, 100), 2)

def handle_mouse_click(event, x, y, flags, param):
    """Handle mouse clicks on the OpenCV window"""
    global dropdown_active, dropdown_selected_idx, selected_id, target_id, selection_mode
    global calibration_active, last_sample_time, last_sample_position
    
    if event == cv2.EVENT_LBUTTONDOWN:
        gui_start_y = FRAME_H
        
        # Check dropdown area
        dropdown_x = 10
        dropdown_y = gui_start_y + 45
        dropdown_width, dropdown_height = 200, 30
        
        if (dropdown_x <= x <= dropdown_x + dropdown_width and
            dropdown_y <= y <= dropdown_y + dropdown_height):
            if selection_mode and dropdown_options:
                dropdown_active = not dropdown_active
        
        # Check dropdown options if active
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
        
        # Check buttons
        button_y = gui_start_y + 80
        button_height = 30
        
        # Auto Select button
        auto_x = 10
        button_width = 120
        if (auto_x <= x <= auto_x + button_width and
            button_y <= y <= button_y + button_height and selection_mode):
            auto_select_id()
        
        # Calibration button
        calib_x = 250
        calib_width = 150
        if (calib_x <= x <= calib_x + calib_width and
            button_y <= y <= button_y + button_height and not selection_mode):
            toggle_calibration()

def auto_select_id():
    """Auto-select the most consistently detected ID"""
    global id_history, selected_id, target_id, selection_mode, dropdown_selected_idx
    if id_history:
        best_id = max(id_history.keys(), key=lambda k: id_history[k])
        selected_id = best_id
        target_id = best_id
        selection_mode = False
        if str(best_id) in dropdown_options:
            dropdown_selected_idx = dropdown_options.index(str(best_id))
        print(f"[gui] Auto-selected ID {best_id} for tracking")

def toggle_calibration():
    """Toggle calibration on/off"""
    global calibration_active, last_sample_time, last_sample_position, cam_pts, map_pts
    
    if calibration_active:
        # Stop calibration and compute final transform
        calibration_active = False
        print("[calib] Calibration stopped")
        if len(cam_pts) >= 3:
            print_final_transform()
        else:
            print(f"[calib] Need ≥3 points, got {len(cam_pts)}")
    else:
        # Start calibration
        calibration_active = True
        last_sample_time = time.time()
        last_sample_position = None
        cam_pts.clear()
        map_pts.clear()
        print(f"[calib] Calibration started - sampling every {SAMPLE_INTERVAL}s")

def print_final_transform():
    """Print the final transformation matrix"""
    if current_transform is not None:
        print("\n" + "="*60)
        print("FINAL TRANSFORMATION MATRIX:")
        print("="*60)
        print("T_MAP_CAM = np.array([")
        for row in current_transform:
            print("  [{: .8f}, {: .8f}, {: .8f}, {: .8f}],".format(*row))
        print("], dtype=float)")
        print("="*60)

def compute_rigid_transform(cam_pts: np.ndarray, map_pts: np.ndarray) -> np.ndarray:
    assert cam_pts.shape == map_pts.shape and cam_pts.shape[0] >= 3
    c_cam = cam_pts.mean(axis=0)
    c_map = map_pts.mean(axis=0)
    A = cam_pts - c_cam
    B = map_pts - c_map
    H = A.T @ B
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    t = c_map - R @ c_cam
    T = np.eye(4)
    T[:3,:3] = R
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

# ─── CLEANUP ──────────────────────────────────────────────────────────────────
def cleanup_and_exit(sig, frame):
    print("\n[exit] Shutting down...")
    pipeline.stop()
    mqttc.loop_stop()
    cv2.destroyAllWindows()
    if calibration_active and len(cam_pts) >= 3:
        print_final_transform()
    sys.exit(0)

signal.signal(signal.SIGINT, cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
cv2.namedWindow("calib", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("calib", handle_mouse_click)
print("** Automatic Calibration **")
print(" - Select a Teleco ID to track")
print(" - Click 'Start Calibration' to begin automatic sampling")
print(" - Move the robot freely, samples taken automatically")
print(" - Click 'Stop Calibration' to get the final matrix")
print(f" - Sample interval: {SAMPLE_INTERVAL}s")
print(f" - Movement threshold: {MIN_MOVEMENT_THRESHOLD}m")
print(f" - Max samples: {MAX_SAMPLES}")
print(" - Press ESC to exit\n")

while True:
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    cf = aligned.get_color_frame()
    df = aligned.get_depth_frame()
    if not cf or not df:
        continue
    
    camera_img = np.asanyarray(cf.get_data())
    depth = np.asanyarray(df.get_data())
    
    img = np.zeros((TOTAL_HEIGHT, FRAME_W, 3), dtype=np.uint8)
    img[:FRAME_H, :FRAME_W] = camera_img
    
    # YOLO detection
    current_frame_ids = set()
    cam_pt = None
    
    if selection_mode:
        # Run full detection to find available IDs
        res = yolo.track(camera_img, conf=CONF_THRESH, iou=IOU_THRESH, 
                        tracker=TRACKER_CFG, persist=True)[0]
        
        for box in res.boxes:
            cls = int(box.cls[0])
            if yolo.names.get(cls,"").lower() != "teleco":
                continue
            rid = box.id
            if rid is None:
                continue
            tid = int(rid[0])
            current_frame_ids.add(tid)
            
            if tid in id_history:
                id_history[tid] += 1
            else:
                id_history[tid] = 1
        
        # Decay history for unseen IDs
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
        
        # Draw all detected telecos
        for box in res.boxes:
            cls = int(box.cls[0])
            if yolo.names.get(cls,"").lower() != "teleco":
                continue
            rid = box.id
            if rid is None:
                continue
            tid = int(rid[0])
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            
            cv2.rectangle(img, (x1,y1),(x2,y2), (255,100,0), 2)
            label = f"ID {tid}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(img, (x1, y1-25), (x1+label_size[0]+5, y1-5), (255,100,0), -1)
            cv2.putText(img, label, (x1+2, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(img, "Available", (x1, y2+20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
    
    else:
        # Tracking mode
        res = yolo.track(camera_img, conf=CONF_THRESH, iou=IOU_THRESH,
                        tracker=TRACKER_CFG, persist=True)[0]
        
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
                
                # Get depth
                patch_size = 5
                patch = depth[max(0,cy-patch_size):cy+patch_size+1,
                             max(0,cx-patch_size):cx+patch_size+1]
                if patch.size:
                    valid_depths = patch[patch > 0]
                    if len(valid_depths) > 0:
                        z = float(np.median(valid_depths)) * depth_scale
                        if z > 0:
                            cam_pt = np.array(rs.rs2_deproject_pixel_to_point(intrinsics, [cx,cy], z))
                
                # Draw tracked robot
                color = (0,255,0)
                cv2.rectangle(img, (x1,y1),(x2,y2), color, 3)
                label = f"ID {tid}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(img, (x1, y1-25), (x1+label_size[0]+5, y1-5), color, -1)
                cv2.putText(img, label, (x1+2, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                
                status = "CALIBRATING" if calibration_active else "TRACKING"
                status_color = (0,255,255) if calibration_active else (0,255,0)
                cv2.putText(img, status, (x1, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                break
    
    # Automatic sampling during calibration
    current_time = time.time()
    if (calibration_active and not selection_mode and cam_pt is not None and
        current_time - last_sample_time >= SAMPLE_INTERVAL):
        
        with lock:
            if latest_map is not None:
                rob_pt = latest_map.copy()
                if has_moved_enough(rob_pt):
                    if add_sample(cam_pt, rob_pt):
                        last_sample_time = current_time
    
    with lock:
        rob_pt = latest_map.copy() if latest_map is not None else None
    
    # Draw GUI
    draw_gui_area(img)
    
    # Status text
    gui_start_y = FRAME_H
    status_y = gui_start_y + 20
    if selection_mode:
        status_text = "MODE: Selection - Choose an ID to track"
        status_color = (0, 255, 255)
    elif calibration_active:
        status_text = f"MODE: Auto Calibrating ID {target_id} (Samples: {len(cam_pts)})"
        status_color = (0, 255, 0)
    else:
        status_text = f"MODE: Ready for calibration - ID {target_id}"
        status_color = (255, 255, 0)
    
    cv2.putText(img, status_text, (10, status_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
    
    # Dropdown and buttons
    if selection_mode:
        dropdown_y = gui_start_y + 45
        draw_dropdown(img, 10, dropdown_y, 200, 30, dropdown_options, dropdown_selected_idx, dropdown_active)
        draw_auto_select_button(img)
    else:
        draw_calibration_button(img)
    
    # Data overlay
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
        txt1 = f"Cam: X:{cam_pt[0]:.2f} Y:{cam_pt[1]:.2f} Z:{cam_pt[2]:.2f}"
    (w1,h1),_ = cv2.getTextSize(txt1, font, fs, th)
    cv2.rectangle(img, (x0-4, y-h1-4), (x0+w1+4, y+4), bg, -1)
    cv2.putText(img, txt1, (x0, y), font, fs, fg, th)
    
    # Robot coordinates
    y += h1 + spacing
    if rob_pt is None:
        txt2 = "Rob: Waiting for MQTT data..."
    else:
        txt2 = f"Rob: X:{rob_pt[0]:.2f} Y:{rob_pt[1]:.2f} Z:{rob_pt[2]:.2f}"
    (w2,h2),_ = cv2.getTextSize(txt2, font, fs, th)
    cv2.rectangle(img, (x0-4, y-h2-4), (x0+w2+4, y+4), bg, -1)
    cv2.putText(img, txt2, (x0, y), font, fs, fg, th)
    
    # Sample info
    y += h2 + spacing
    if calibration_active:
        next_sample = max(0, SAMPLE_INTERVAL - (current_time - last_sample_time))
        txt3 = f"Samples: {len(cam_pts)}/{MAX_SAMPLES} | Next: {next_sample:.1f}s"
    else:
        txt3 = f"Samples collected: {len(cam_pts)}"
    (w3,h3),_ = cv2.getTextSize(txt3, font, fs, th)
    cv2.rectangle(img, (x0-4, y-h3-4), (x0+w3+4, y+4), bg, -1)
    cv2.putText(img, txt3, (x0, y), font, fs, fg, th)
    
    cv2.imshow("calib", img)
    
    key = cv2.waitKey(1) & 0xFF
    
    # SPACE = toggle calibration (alternative to button)
    if key == ord(' ') and not selection_mode:
        toggle_calibration()
    
    # ESC = exit
    if key == 27:
        cleanup_and_exit(None, None)