#!/usr/bin/env python3

import cv2
import numpy as np
import torch
import csv
import os
from ultralytics import YOLO
from ultralytics.trackers import BYTETracker, BOTSORT
from ultralytics.engine.results import Boxes
from ultralytics.utils import yaml_load, IterableSimpleNamespace
from pathlib import Path

# ========== Load Tracker Config ==========
CONFIG_PATH = str(Path.home() / "clearpath_ws/src/vision_tracking_pose/vision_tracking_pose/botsort.yaml")  # Change to botsort.yaml for BoT-SORT
cfg = yaml_load(CONFIG_PATH)
args = IterableSimpleNamespace(**cfg)

# Set Tracker
tracker_name = "BoT-SORT" if cfg.get("tracker_type") == "botsort" else "ByteTrack"
print(f"\n Using tracker: {tracker_name}\n")

# ========== Load YOLOv8 Pose Model ==========
model = YOLO("/home/hiba/clearpath_ws/src/vision_tracking_pose/vision_tracking_pose/yolov8n-pose.pt")

# ========== Initialize Tracker ==========
if tracker_name == "BoT-SORT":
    tracker = BOTSORT(args=args, frame_rate=30)
else:
    tracker = BYTETracker(args=args, frame_rate=30)

# ========== Open Video ==========
cap = cv2.VideoCapture("/home/hiba/clearpath_ws/src/vision_tracking_pose/vision_tracking_pose/palace.mp4")

# ========== Prepare CSV Logging ==========
csv_filename = f"/home/hiba/clearpath_ws/src/vision_tracking_pose/vision_tracking_pose/tracking_results_{tracker_name.lower().replace('-', '')}.csv"
is_new_file = not os.path.exists(csv_filename)

if is_new_file:
    with open(csv_filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame", "track_id", "x1", "y1", "x2", "y2", "conf", "class"])

# ========== Tracking Loop ==========
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLO Pose Inference
    results = model(frame)[0]

    if results.boxes is None or len(results.boxes) == 0:
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    classes = results.boxes.cls.cpu().numpy()

    detections_array = np.column_stack((boxes, confs, classes)).astype(np.float32)
    det_boxes = Boxes(detections_array, orig_shape=frame.shape[:2])
    tracks = tracker.update(det_boxes, frame)

    # Get current frame number
    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

    # ========== Save to CSV ==========
    with open(csv_filename, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        for track in tracks:
            x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
            conf = float(track[5]) if len(track) > 5 else 0.0
            cls = int(track[6]) if len(track) > 6 else -1
            writer.writerow([frame_num, track_id, x1, y1, x2, y2, conf, cls])

            # Draw Box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # ========== Draw Keypoints ==========
    if results.keypoints is not None:
        for keypoints in results.keypoints.xy:
            for kp in keypoints:
                x, y = int(kp[0]), int(kp[1])
                cv2.circle(frame, (x, y), 4, (0, 255, 255), -1)

    # ========== Show Frame ==========
    cv2.imshow("Tracking with Pose", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"\n Tracking finished. Results saved to: {csv_filename}\n")

