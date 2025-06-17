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
        # ─── TK window setup ───
        self.root = tk.Tk()
        self.root.title("Robot⇆Human Tracker")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        signal.signal(signal.SIGINT, lambda s, f: self._cleanup_and_exit())

        # Variables & IDs
        self.robot_var = tk.StringVar()
        self.human_var = tk.StringVar()
        self.last_ids  = []

        # Tracking & smoothing
        self.tracks     = {}
        self.local_map  = {}
        self.free_ids   = []
        self.next_local = 1
        self.alpha      = 0.7

        # Toggle coords
        self.show_coords = True

        # Load YOLOv8
        self.model = YOLO('/home/commu/Desktop/human_detector_ws/models/best_yolo11s.pt')

        # RealSense init
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile      = self.pipeline.start(cfg)
        self.align   = rs.align(rs.stream.color)
        self.intr    = profile.get_stream(rs.stream.depth) \
                               .as_video_stream_profile() \
                               .get_intrinsics()

        # GUI
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        ctrl = tk.Frame(self.root)
        ctrl.pack(fill=tk.X, pady=5)
        tk.Label(ctrl, text="First ID:").pack(side=tk.LEFT)
        self.robot_menu = ttk.OptionMenu(ctrl, self.robot_var, "")
        self.robot_menu.pack(side=tk.LEFT, padx=5)
        tk.Label(ctrl, text="Second ID:").pack(side=tk.LEFT)
        self.human_menu = ttk.OptionMenu(ctrl, self.human_var, "")
        self.human_menu.pack(side=tk.LEFT, padx=5)
        self.toggle_button = tk.Button(ctrl, text="Hide Coords", command=self._toggle_coords)
        self.toggle_button.pack(side=tk.LEFT, padx=5)

        self.dist_label = tk.Label(self.root, text="Distance: N/A", font=("Arial", 14))
        self.dist_label.pack(pady=5)

        # Start loop
        self.update_frame()
        self.root.mainloop()

    def _toggle_coords(self):
        self.show_coords = not self.show_coords
        self.toggle_button.config(text="Show Coords" if not self.show_coords else "Hide Coords")

    def _cleanup_and_exit(self):
        self.on_closing()
        sys.exit(0)

    def update_frame(self):
        # Grab frames
        frames  = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color   = aligned.get_color_frame()
        depth   = aligned.get_depth_frame()
        img     = np.asanyarray(color.get_data())

        # YOLO tracking
        #   conf      = 0.6    # Detection confidence threshold: 
        #                      #   ↑ higher → fewer false positives, but may miss small/occluded objects
        #                      #   ↓ lower  → more detections, but more noise
        #
        #   iou       = 0.7    # NMS IoU threshold:
        #                      #   ↓ lower → stricter merging, less box overlap
        #                      #   ↑ higher→ allow closer boxes, may keep duplicates

        # results = self.model.track(img, conf=0.6, iou=0.7,
        #                            tracker='bytetrack.yaml', persist=True)[0]
        results = self.model.track(img, conf=0.6, iou=0.8,
                            tracker='bytetrack.yaml', persist=True)[0]

        current, current_ids = {}, set()

        for box in results.boxes:
            raw_id = box.id
            if raw_id is None:
                continue
            tid = int(raw_id[0]); current_ids.add(tid)

            # Class name
            cls_idx  = int(box.cls[0])
            cls_name = self.model.names.get(cls_idx, "").capitalize() or "Object"

            # Box coords and center
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cx, cy        = (x1 + x2)//2, (y1 + y2)//2

            # Depth + deproject
            z = depth.get_distance(cx, cy)
            if z == 0:
                continue
            X, Y, Z = rs.rs2_deproject_pixel_to_point(self.intr, [cx, cy], z)

            # EMA smoothing
            if tid in self.tracks:
                prev = self.tracks[tid]['smoothed']
                x1 = int(self.alpha*x1 + (1-self.alpha)*prev['x1'])
                y1 = int(self.alpha*y1 + (1-self.alpha)*prev['y1'])
                x2 = int(self.alpha*x2 + (1-self.alpha)*prev['x2'])
                y2 = int(self.alpha*y2 + (1-self.alpha)*prev['y2'])
                Z  =     self.alpha*Z  + (1-self.alpha)*prev['z']

            # Store
            self.tracks.setdefault(tid, {})['smoothed'] = {
                'x1':x1,'y1':y1,'x2':x2,'y2':y2,'z':Z
            }
            current[tid] = {'box':(x1,y1,x2,y2),'world':(X,Y,Z),'class':cls_name}

        # Cleanup departed
        departed = set(self.tracks) - current_ids
        for tid in departed:
            if tid in self.local_map:
                self.free_ids.append(self.local_map[tid])
                del self.local_map[tid]
            del self.tracks[tid]

        # Reset if none
        if not current_ids:
            self.local_map.clear(); self.free_ids.clear()
            self.tracks.clear(); self.next_local = 1

        # Assign local IDs
        for tid in current_ids:
            if tid not in self.local_map:
                lid = self.free_ids.pop(0) if self.free_ids else self.next_local
                self.local_map[tid] = lid
                if lid == self.next_local:
                    self.next_local += 1

        # Build objects
        objects = {self.local_map[tid]:data for tid,data in current.items()}

        # Update menus
        ids = sorted(objects)
        if ids != self.last_ids:
            self._update_menu(self.robot_menu, self.robot_var, ids)
            self._update_menu(self.human_menu, self.human_var, ids)
            self.last_ids = ids

        # --- Drawing ---
        for lid, obj in objects.items():
            x1,y1,x2,y2 = obj['box']
            wx,wy,wz    = obj['world']
            cls_name    = obj['class']

            # box color
            if str(lid) == self.robot_var.get():   col=(255,128,0)
            elif str(lid) == self.human_var.get(): col=(0,128,255)
            else:                                  col=(200,200,200)

            # draw bbox
            cv2.rectangle(img, (x1,y1), (x2,y2), col, 2)

            # ID label above box
            id_txt = f"ID {lid}"
            (tw,th),_ = cv2.getTextSize(id_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1-th-6), (x1+tw+6, y1), col, cv2.FILLED)
            cv2.putText(img, id_txt, (x1+3, y1-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)

            # NAME box inside top-left
            name_font, name_fs, name_thk = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            (nw, nh), _ = cv2.getTextSize(cls_name, name_font, name_fs, name_thk)
            nx1, ny1 = x1+2, y1+2
            nx2, ny2 = nx1+nw+4, ny1+nh+4
            cv2.rectangle(img, (nx1,ny1), (nx2,ny2), col, cv2.FILLED)
            cv2.putText(img, cls_name, (nx1+2, ny2-2),
                        name_font, name_fs, (0,0,0), name_thk)

            # COORDS box directly below name box
            if self.show_coords:
                coord_font, fs, thk = cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                labels = [f"X:{wx:.2f}m", f"Y:{wy:.2f}m", f"Z:{wz:.2f}m"]
                # compute width of full coords string
                total_w = 0
                spac = 8
                sizes = []
                for lbl in labels:
                    (w,h),_ = cv2.getTextSize(lbl, coord_font, fs, thk)
                    sizes.append((w,h))
                    total_w += w
                total_w += spac*(len(labels)-1)
                ch = max(h for w,h in sizes)

                cx1 = nx1
                cy1 = ny2 + 2
                cx2 = cx1 + total_w + 4
                cy2 = cy1 + ch + 4
                # background box: dark gray
                cv2.rectangle(img, (cx1,cy1), (cx2,cy2), (50,50,50), cv2.FILLED)

                # draw each coord
                tx = cx1 + 2
                ty = cy2 - 2
                for (w,h), lbl in zip(sizes, labels):
                    color = (0,255,0) if lbl.startswith("Z:") else (255,255,255)
                    cv2.putText(img, lbl, (tx, ty), coord_font, fs, color, thk)
                    tx += w + spac

        # Distance display
        rv, hv = self.robot_var.get(), self.human_var.get()
        if rv.isdigit() and hv.isdigit():
            r,h = int(rv), int(hv)
            if r in objects and h in objects:
                wx,wy,wz = objects[r]['world']
                hx,hy,hz = objects[h]['world']
                d = math.sqrt((wx-hx)**2 + (wy-hy)**2 + (wz-hz)**2)
                self.dist_label.config(text=f"Distance: {d:.2f} m")
            else:
                self.dist_label.config(text="Distance: N/A")
        else:
            self.dist_label.config(text="Distance: N/A")

        # Render to Tk
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil  = Image.fromarray(img_rgb)
        imgtk   = ImageTk.PhotoImage(image=im_pil)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        self.root.after(30, self.update_frame)

    def _update_menu(self, menu_widget, var, choices):
        menu = menu_widget["menu"]
        menu.delete(0, "end")
        for c in choices:
            menu.add_command(label=str(c), command=lambda v=c: var.set(str(v)))
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
