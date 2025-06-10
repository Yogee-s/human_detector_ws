#!/usr/bin/env python3
import signal
import sys
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import math
import cv2

class App:
    def __init__(self):
        # ——— TK window setup ———
        self.root = tk.Tk()
        self.root.title("Robot⇆Human Tracker")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # allow Ctrl+C to close
        signal.signal(signal.SIGINT, lambda s, f: self._cleanup_and_exit())

        # Variables & last IDs
        self.robot_var = tk.StringVar()
        self.human_var = tk.StringVar()
        self.last_ids = []  # for menu updates

        # ** Tracking and smoothing storage **
        self.tracks = {}    # track_id -> {'smoothed': {...}}
        self.local_map = {} # track_id -> local_id
        self.free_ids = []  # reusable local IDs
        self.next_local = 1
        self.alpha = 0.7    # EMA smoothing factor

        # Load YOLOv8 model
        self.model = YOLO(
            '/home/commu/Desktop/human_detector_ws/src/human_detector/models/best_yolo11s.pt'
        )

        # RealSense init
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.intr = profile.get_stream(rs.stream.depth) \
                       .as_video_stream_profile() \
                       .get_intrinsics()

        # GUI elements
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        ctrl = tk.Frame(self.root)
        ctrl.pack(fill=tk.X, pady=5)
        tk.Label(ctrl, text="Robot ID:").pack(side=tk.LEFT)
        self.robot_menu = ttk.OptionMenu(ctrl, self.robot_var, "")
        self.robot_menu.pack(side=tk.LEFT, padx=5)
        tk.Label(ctrl, text="Human ID:").pack(side=tk.LEFT)
        self.human_menu = ttk.OptionMenu(ctrl, self.human_var, "")
        self.human_menu.pack(side=tk.LEFT, padx=5)

        self.dist_label = tk.Label(self.root, text="Distance: N/A",
                                   font=("Arial", 14))
        self.dist_label.pack(pady=5)

        # Begin loop
        self.update_frame()
        self.root.mainloop()

    def _cleanup_and_exit(self):
        self.on_closing()
        sys.exit(0)

    def update_frame(self):
        # Acquire aligned frames
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        img = np.asanyarray(color.get_data())

        # YOLOv8 tracking
        # results = self.model.track(
        #     img, conf=0.3, iou=0.6,
        #     tracker='bytetrack.yaml', persist=True
        # )[0]
        results = self.model.track(
            img, conf=0.6, iou=0.7,
            tracker='bytetrack.yaml', persist=True
        )[0]

        current = {}
        current_ids = set()

        # Process each detection
        for box in results.boxes:
            raw_id = box.id
            if raw_id is None:
                continue
            tid = int(raw_id[0])
            current_ids.add(tid)

            # Classification label
            cls_idx = int(box.cls[0])
            raw_cls = self.model.names.get(cls_idx, "")

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            z = depth.get_distance(cx, cy)
            if z == 0:
                continue
            X, Y, Z = rs.rs2_deproject_pixel_to_point(self.intr, [cx, cy], z)

            # EMA smoothing
            if tid in self.tracks:
                prev = self.tracks[tid]['smoothed']
                x1 = int(self.alpha * x1 + (1 - self.alpha) * prev['x1'])
                y1 = int(self.alpha * y1 + (1 - self.alpha) * prev['y1'])
                x2 = int(self.alpha * x2 + (1 - self.alpha) * prev['x2'])
                y2 = int(self.alpha * y2 + (1 - self.alpha) * prev['y2'])
                Z  =     self.alpha * Z  + (1 - self.alpha) * prev['z']

            # store
            self.tracks.setdefault(tid, {})
            self.tracks[tid]['smoothed'] = {
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'z': Z
            }
            current[tid] = {
                'box':   (x1, y1, x2, y2),
                'world': (X, Y, Z),
                'class': raw_cls
            }

        # Clean up departed
        departed = set(self.tracks.keys()) - current_ids
        for tid in departed:
            if tid in self.local_map:
                self.free_ids.append(self.local_map[tid])
                del self.local_map[tid]
            del self.tracks[tid]

        # If no subjects remain, reset all IDs
        if not current_ids:
            self.local_map.clear()
            self.free_ids.clear()
            self.tracks.clear()
            self.next_local = 1

        # Assign local IDs
        for tid in current_ids:
            if tid not in self.local_map:
                if self.free_ids:
                    lid = self.free_ids.pop(0)
                else:
                    lid = self.next_local
                    self.next_local += 1
                self.local_map[tid] = lid

        # Build objects dict with local_ids
        objects = {}
        for tid, data in current.items():
            lid = self.local_map[tid]
            objects[lid] = {
                'box':   data['box'],
                'world': data['world'],
                'class': data['class']
            }

        ids = sorted(objects.keys())
        # update menus if changed
        if ids != self.last_ids:
            self._update_menu(self.robot_menu, self.robot_var, ids)
            self._update_menu(self.human_menu, self.human_var, ids)
            self.last_ids = ids

        # Draw
        for lid, obj in objects.items():
            x1, y1, x2, y2 = obj['box']
            # color coding
            if str(lid) == self.robot_var.get():
                col = (255, 128, 0)
            elif str(lid) == self.human_var.get():
                col = (0, 128, 255)
            else:
                col = (200, 200, 200)

            cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)

            # Build label text
            cls = obj.get('class', '').lower()
            if cls in ('person', 'human'):
                prefix = 'Human'
            elif cls:
                prefix = cls.capitalize()
            else:
                prefix = 'ID'
            text = f"{prefix} {lid}"

            (tw, th), _ = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            # filled background
            cv2.rectangle(
                img,
                (x1, y1 - th - 6),
                (x1 + tw + 6, y1),
                col,
                cv2.FILLED
            )
            cv2.putText(
                img, text, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (0, 0, 0), 2
            )

        # Compute and show distance
        rv, hv = self.robot_var.get(), self.human_var.get()
        if rv.isdigit() and hv.isdigit():
            ri, hi = int(rv), int(hv)
            if ri in objects and hi in objects:
                wx, wy, wz = objects[ri]['world']
                hx, hy, hz = objects[hi]['world']
                d = math.sqrt(
                    (wx - hx)**2 + (wy - hy)**2 + (wz - hz)**2
                )
                self.dist_label.config(text=f"Distance: {d:.2f} m")
            else:
                self.dist_label.config(text="Distance: N/A")
        else:
            self.dist_label.config(text="Distance: N/A")

        # render
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rgb)
        imgtk = ImageTk.PhotoImage(image=im_pil)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        # next
        self.root.after(30, self.update_frame)

    def _update_menu(self, menu_widget, var, choices):
        m = menu_widget["menu"]
        m.delete(0, "end")
        for c in choices:
            m.add_command(
                label=str(c),
                command=lambda v=c: var.set(str(v))
            )
        if var.get() not in map(str, choices):
            var.set("")

    def on_closing(self):
        try:
            self.pipeline.stop()
        except:
            pass
        self.root.destroy()

if __name__ == "__main__":
    App()
