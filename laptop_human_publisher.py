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
        # ——— 1) build the TK window first
        self.root = tk.Tk()
        self.root.title("Robot⇆Human Tracker")
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # clean exit on Ctrl+C
        signal.signal(signal.SIGINT, lambda s,f: self._cleanup_and_exit())

        # now safe to make StringVars
        self.robot_var = tk.StringVar()
        self.human_var = tk.StringVar()
        self.last_ids = []            # remember last ID list

        # load YOLO
        self.model = YOLO('/home/yogee/Desktop/human_detector_ws/src/human_detector/models/best_v2_yolov11n.pt')

        # RealSense init
        self.pipeline = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipeline.start(cfg)
        self.align = rs.align(rs.stream.color)
        self.intr = profile.get_stream(rs.stream.depth).as_video_stream_profile().get_intrinsics()

        # GUI layout
        self.video_label = tk.Label(self.root)
        self.video_label.pack()

        ctrl = tk.Frame(self.root); ctrl.pack(fill=tk.X, pady=5)
        tk.Label(ctrl, text="Robot ID:").pack(side=tk.LEFT)
        self.robot_menu = ttk.OptionMenu(ctrl, self.robot_var, "")
        self.robot_menu.pack(side=tk.LEFT, padx=5)
        tk.Label(ctrl, text="Human ID:").pack(side=tk.LEFT)
        self.human_menu = ttk.OptionMenu(ctrl, self.human_var, "")
        self.human_menu.pack(side=tk.LEFT, padx=5)

        self.dist_label = tk.Label(self.root, text="Distance: N/A", font=("Arial",14))
        self.dist_label.pack(pady=5)

        # start the loop
        self.update_frame()
        self.root.mainloop()

    def _cleanup_and_exit(self):
        self.on_closing()
        sys.exit(0)

    def update_frame(self):
        # grab aligned frames
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color = aligned.get_color_frame()
        depth = aligned.get_depth_frame()
        img = np.asanyarray(color.get_data())

        # detect per‐frame, assign IDs 1..n
        dets = []
        results = self.model(img)[0]
        for idx, (*box, conf, cls) in enumerate(results.boxes.data.tolist(), start=1):
            if conf < 0.3:               # lower threshold
                continue
            x1,y1,x2,y2 = map(int, box)
            cx,cy = (x1+x2)//2, (y1+y2)//2
            z = depth.get_distance(cx,cy)
            if z==0:
                continue
            X,Y,Z = rs.rs2_deproject_pixel_to_point(self.intr, [cx,cy], z)
            dets.append({
                'id':       idx,
                'box':      (x1,y1,x2,y2),
                'cls':      self.model.names[int(cls)],
                'world':    (X,Y,Z)
            })

        # build fresh object dict
        objects = {d['id']:d for d in dets}
        ids = list(objects.keys())

        # only rebuild menus if the set of IDs changed
        if ids != self.last_ids:
            self._update_menu(self.robot_menu, self.robot_var, ids)
            self._update_menu(self.human_menu, self.human_var, ids)
            self.last_ids = ids

        # draw boxes + labels
        for oid,obj in objects.items():
            x1,y1,x2,y2 = obj['box']
            color = (0,255,0) if str(oid)==self.robot_var.get() else \
                    (255,0,0) if str(oid)==self.human_var.get() else \
                    (0,0,255)
            cv2.rectangle(img, (x1,y1),(x2,y2), color, 2)
            text = f"{obj['cls']}:{oid}"
            ty = y1+20 if y1<10 else y1-10
            cv2.putText(img, text, (x1,ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # compute 3D distance if both selected
        rv, hv = self.robot_var.get(), self.human_var.get()
        if rv.isdigit() and hv.isdigit():
            ri, hi = int(rv), int(hv)
            if ri in objects and hi in objects:
                wx,wy,wz = objects[ri]['world']
                hx,hy,hz = objects[hi]['world']
                d = math.sqrt((wx-hx)**2 + (wy-hy)**2 + (wz-hz)**2)
                self.dist_label.config(text=f"Distance: {d:.2f} m")
            else:
                self.dist_label.config(text="Distance: N/A")
        else:
            self.dist_label.config(text="Distance: N/A")

        # render to Tk
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        im_pil  = Image.fromarray(img_rgb)
        imgtk   = ImageTk.PhotoImage(image=im_pil)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk)

        # schedule next
        self.root.after(30, self.update_frame)

    def _update_menu(self, menu_widget, var, choices):
        m = menu_widget["menu"]
        m.delete(0, "end")
        for c in choices:
            m.add_command(label=str(c), command=lambda v=c: var.set(str(v)))
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
