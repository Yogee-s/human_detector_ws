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
MQTT_ADDRESS  = "mittsu-talk.jp"
MQTT_PORT     = 443
MQTT_PATH     = "/mosmos-test2/ws/"
MQTT_TOPIC    = "/topic/testing/testroom/ROVER-001/info"

YOLO_MODEL    = "/home/commu/Desktop/human_detector_ws/models/best_yolo11m.pt"
FRAME_W, FRAME_H = 640, 480
CONF_THRESH   = 0.4
IOU_THRESH    = 0.5
TRACKER_CFG   = "bytetrack.yaml"
# ───────────────────────────────────────────────────────────────────────────────

logging.getLogger("ultralytics").setLevel(logging.ERROR)

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────────
cam_pts     = []     # list of np.array([Xc,Yc,Zc])
map_pts     = []     # list of np.array([x_map,y_map,z_map])
lock        = threading.Lock()
latest_map  = None   # last robot pose from MQTT
target_id   = None   # locked YOLO track ID

# ─── KABSCH SOLVER ────────────────────────────────────────────────────────────
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
            latest_map = np.array([x,y,z],dtype=float)
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
    print("\n[exit] computing final transform…")
    pipeline.stop()
    mqttc.loop_stop()
    if len(cam_pts) >= 3:
        C = np.stack(cam_pts, axis=0)
        M = np.stack(map_pts, axis=0)
        T = compute_rigid_transform(C, M)
        # print in copy-pasteable form:
        print("\nT_MAP_CAM = np.array([")
        for row in T:
            print("    [{: .8f}, {: .8f}, {: .8f}, {: .8f}],".format(*row))
        print("], dtype=float)")
    else:
        print(f"need ≥3 points, got {len(cam_pts)}")
    sys.exit(0)

signal.signal(signal.SIGINT,  cleanup_and_exit)
signal.signal(signal.SIGTERM, cleanup_and_exit)

# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
cv2.namedWindow("calib", cv2.WINDOW_NORMAL)

print("** Calibration running **")
print(" - hit [SPACE] to record a pair")
print(" - press Ctrl+C or Esc to finish & compute the transform\n")

while True:
    frames = pipeline.wait_for_frames()
    aligned = align.process(frames)
    cf = aligned.get_color_frame(); df = aligned.get_depth_frame()
    if not cf or not df:
        continue

    img   = np.asanyarray(cf.get_data())
    depth = np.asanyarray(df.get_data())

    # YOLO + tracker
    res = yolo.track(img,
                     conf=CONF_THRESH,
                     iou=IOU_THRESH,
                     tracker=TRACKER_CFG,
                     persist=True)[0]

    # draw Teleco boxes & lock hint
    for box in res.boxes:
        cls = int(box.cls[0])
        if yolo.names.get(cls,"").lower() != "teleco":
            continue
        rid = box.id
        if rid is None:
            continue
        tid = int(rid[0])
        x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
        color = (0,255,0) if (target_id is None or tid==target_id) else (100,100,100)
        cv2.rectangle(img, (x1,y1),(x2,y2), color, 2)
        cv2.putText(img, f"ID {tid}", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if target_id is None:
            cv2.putText(img, "(press SPACE to lock)", (x1, y2+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)

    # compute current cam_pt
    cam_pt = None
    for box in res.boxes:
        cls = int(box.cls[0])
        if yolo.names.get(cls,"").lower() != "teleco":
            continue
        rid = box.id
        if rid is None:
            continue
        tid = int(rid[0])
        if target_id is None or tid==target_id:
            x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
            cx,cy = (x1+x2)//2, (y1+y2)//2
            patch = depth[max(0,cy-2):cy+3, max(0,cx-2):cx+3]
            if patch.size:
                z = float(np.median(patch)) * depth_scale
                if z>0:
                    cam_pt = rs.rs2_deproject_pixel_to_point(intrinsics, [cx,cy], z)
            break

    with lock:
        rob_pt = latest_map.copy() if latest_map is not None else None

    # overlay: dark-grey bg + white text
    y = 30; x0 = 10
    font = cv2.FONT_HERSHEY_SIMPLEX; fs, th = 0.6, 2
    bg = (50,50,50); fg = (255,255,255)

    # if no cam_pt yet, show how many pairs have been recorded
    if cam_pt is None:
        txt1 = f"Pairs recorded: {len(cam_pts)}"
    else:
        txt1 = f"Cam → X:{cam_pt[0]:.2f}  Y:{cam_pt[1]:.2f}  Z:{cam_pt[2]:.2f}"
    (w1,h1),_ = cv2.getTextSize(txt1, font, fs, th)
    cv2.rectangle(img, (x0-4, y-h1-4), (x0+w1+4, y+4), bg, -1)
    cv2.putText(img, txt1, (x0, y), font, fs, fg, th)

    # second line: robot
    y += h1 + 12
    if rob_pt is None:
        txt2 = "Rob → no data yet"
    else:
        txt2 = f"Rob → X:{rob_pt[0]:.2f}  Y:{rob_pt[1]:.2f}  Z:{rob_pt[2]:.2f}"
    (w2,h2),_ = cv2.getTextSize(txt2, font, fs, th)
    cv2.rectangle(img, (x0-4, y-h2-4), (x0+w2+4, y+4), bg, -1)
    cv2.putText(img, txt2, (x0, y), font, fs, fg, th)

    cv2.imshow("calib", img)
    key = cv2.waitKey(1) & 0xFF

    # SPACE = record one pair
    if key == ord(' '):
        print("[input] SPACE pressed")
        picked = None
        for box in res.boxes:
            cls = int(box.cls[0])
            if yolo.names.get(cls,"").lower() != "teleco":
                continue
            rid = box.id
            if rid is None:
                continue
            tid = int(rid[0])
            if target_id is None or tid==target_id:
                x1,y1,x2,y2 = map(int, box.xyxy[0].tolist())
                cx,cy = (x1+x2)//2, (y1+y2)//2
                patch = depth[max(0,cy-2):cy+3, max(0,cx-2):cx+3]
                if patch.size:
                    z = float(np.median(patch)) * depth_scale
                    if z>0:
                        picked = (tid, np.array(
                            rs.rs2_deproject_pixel_to_point(intrinsics, [cx,cy], z),
                            dtype=float))
                break

        with lock:
            if picked and latest_map is not None:
                tid, cam3 = picked
                if target_id is None:
                    target_id = tid
                    print(f"[lock] using tracker ID {target_id}")
                mp_copy = latest_map.copy()
                cam_pts.append(cam3)
                map_pts.append(mp_copy)
                print(f"[recorded] pair #{len(cam_pts)}: cam {cam3.round(3)} ↔ map {mp_copy.round(3)}")
            else:
                print("[warning] skipped—no detection or no robot pose")

    # ESC = exit + compute
    if key == 27:
        cleanup_and_exit(None, None)
