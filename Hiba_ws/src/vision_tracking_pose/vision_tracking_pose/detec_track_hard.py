#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge
from rclpy.qos import qos_profile_sensor_data

import cv2
import numpy as np
import torch
import time
from ultralytics import YOLO
from ultralytics.trackers import BYTETracker
from ultralytics.engine.results import Boxes
from ultralytics.utils import yaml_load, IterableSimpleNamespace
from pathlib import Path
from vision_tracking_pose_msgs.msg import Detection, DetectionArray, KeyPoint2DArray, KeyPoint2D


class OptimizedTrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.bridge = CvBridge()
        self.frame_count = 0
        self.fps_log = []
        self.last_tracks = []
        self.missed_frames = 0

        config_path = str(Path.home() / "clearpath_ws/src/vision_tracking_pose/vision_tracking_pose/bytetrack.yaml")
        cfg = yaml_load(config_path)
        args = IterableSimpleNamespace(**cfg)

        self.model = YOLO("clearpath_ws/src/vision_tracking_pose/vision_tracking_pose/yolov8n-pose.pt").to("cuda")
        self.get_logger().info(f" CUDA enabled: {torch.cuda.is_available()} on {torch.cuda.get_device_name(0)}")

        self.tracker = BYTETracker(args=args, frame_rate=13)

        self.image_sub = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        self.track_pub = self.create_publisher(DetectionArray, 'tracking', 10)
        self.get_logger().info(" Optimized Tracking Node Initialized")

    def image_callback(self, msg):
        start_total = time.time()

        original_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        resized_frame = original_frame
        resized_h, resized_w = resized_frame.shape[:2]
        scale_x = scale_y = 1.0

        results = self.model(resized_frame, imgsz=resized_h)[0]

        tracked_msg = DetectionArray()
        tracked_msg.header = msg.header
        tracked_msg.header.frame_id = "camera_color_optical_frame"

        boxes = results.boxes.xyxy.cpu().numpy()
        confs = results.boxes.conf.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        valid = confs > 0.2
        detections_array = np.column_stack((boxes, confs, classes))[valid]

        if detections_array.shape[0] > 0:
            det_boxes = Boxes(detections_array, orig_shape=resized_frame.shape[:2])
            tracks = self.tracker.update(det_boxes, resized_frame)
        else:
            self.get_logger().info("No detections above confidence threshold.")
            tracks = []

        if len(tracks) == 0 and len(self.last_tracks) > 0:
            tracks = self.last_tracks
            self.missed_frames += 1
        else:
            self.last_tracks = tracks
            self.missed_frames = 0

        all_keypoints = results.keypoints.xy.cpu().numpy() if results.keypoints is not None else []

        for track in tracks:
            x1, y1, x2, y2 = [int(coord) for coord in track[:4]]
            track_id = int(track[4])
            score = float(track[5]) if len(track) > 5 else 0.0
            cls = int(track[6]) if len(track) > 6 else 0

            x1_orig = int(x1 * scale_x)
            y1_orig = int(y1 * scale_y)
            x2_orig = int(x2 * scale_x)
            y2_orig = int(y2 * scale_y)

            det = Detection()
            det.id = str(track_id)
            det.bbox.center.position.x = float((x1_orig + x2_orig) / 2)  
            det.bbox.center.position.y = float((y1_orig + y2_orig) / 2)
            
            det.bbox.size.x = float(x2_orig - x1_orig)
            det.bbox.size.y = float(y2_orig - y1_orig)
            det.class_id = cls
            det.score = score

            matched = False
            for keypoints in all_keypoints:
                if (
                    keypoints is None or
                    not isinstance(keypoints, (list, np.ndarray)) or
                    len(keypoints) == 0 or
                    all(len(kp) < 2 or (kp[0] == 0 and kp[1] == 0) for kp in keypoints)
                ):
                    self.get_logger().warn("Skipping invalid or empty keypoint entry.")
                    continue

                try:
                    kp_min_x = min(kp[0] for kp in keypoints)
                    kp_max_x = max(kp[0] for kp in keypoints)
                    kp_min_y = min(kp[1] for kp in keypoints)
                    kp_max_y = max(kp[1] for kp in keypoints)
                except Exception as e:
                    self.get_logger().warn(f"Error computing keypoint bounds: {e}")
                    continue

                bbox_cx = (x1 + x2) / 2
                bbox_cy = (y1 + y2) / 2

                if kp_min_x <= bbox_cx <= kp_max_x and kp_min_y <= bbox_cy <= kp_max_y:
                    kp_array = KeyPoint2DArray()
                    for idx, kp in enumerate(keypoints):
                        try:
                            x, y = float(kp[0] * scale_x), float(kp[1] * scale_y)
                            kp_msg = KeyPoint2D()
                            kp_msg.id = idx + 1
                            kp_msg.point.x = x
                            kp_msg.point.y = y
                            kp_msg.score = 1.0
                            kp_array.data.append(kp_msg)
                        except Exception:
                            self.get_logger().warn("Invalid keypoint skipped.")
                    det.keypoints = kp_array
                    matched = True
                    break

            tracked_msg.detections.append(det)

        self.track_pub.publish(tracked_msg)

        torch.cuda.empty_cache()

        fps = 1.0 / (time.time() - start_total)
        self.fps_log.append(fps)
        if len(self.fps_log) > 30:
            self.fps_log.pop(0)
        avg_fps = sum(self.fps_log) / len(self.fps_log)
        self.get_logger().info(f" Frame Time: {1000/fps:.2f} ms | FPS: {fps:.2f} (avg: {avg_fps:.2f})")


def main(args=None):
    rclpy.init(args=args)
    node = OptimizedTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
