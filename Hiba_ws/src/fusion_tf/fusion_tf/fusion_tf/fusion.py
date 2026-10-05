#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Image, CameraInfo
from vision_tracking_pose_msgs.msg import DetectionArray
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf2_ros
import tf2_sensor_msgs.tf2_sensor_msgs as tf2_sensor_msgs
import sensor_msgs_py.point_cloud2 as pc2
import matplotlib.pyplot as plt
import statistics
import random


def rectContains(rect, pt, w, h, shrink_factor=0.3):
    x1 = int(rect[0] * w - rect[2] * w * 0.5 * (1 - shrink_factor))
    y1 = int(rect[1] * h - rect[3] * h * 0.5 * (1 - shrink_factor))
    x2 = int(rect[0] * w + rect[2] * w * 0.5 * (1 - shrink_factor))
    y2 = int(rect[1] * h + rect[3] * h * 0.5 * (1 - shrink_factor))
    return x1 < pt[0] < x2 and y1 < pt[1] < y2

def filter_outliers(distances):
    if len(distances) < 2:
        return distances
    mu = statistics.mean(distances)
    std = statistics.stdev(distances)
    return [x for x in distances if abs(x - mu) < std]

def get_best_distance(distances, technique="closest"):
    if not distances:
        return None
    if technique == "closest":
        return min(distances)
    elif technique == "average":
        return statistics.mean(distances)
    elif technique == "random":
        return random.choice(distances)
    else:
        return statistics.median(sorted(distances))


class LidarCameraFusionNode(Node):
    def __init__(self):
        super().__init__('lidar_camera_fusion_node')

        self.bridge = CvBridge()
        self.fx = self.fy = self.cx = self.cy = None
        self.image = None
        self.detections = []

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, '/a200_0846/sensors/camera_0/color/image', self.image_callback, 10)
        self.create_subscription(CameraInfo, '/a200_0846/sensors/camera_0/color/camera_info', self.camera_info_callback, 10)
        self.create_subscription(PointCloud2, '/cloud', self.pointcloud_callback, 10)
        self.create_subscription(DetectionArray, '/tracking', self.detections_callback, 10)

        self.publisher_debug = self.create_publisher(Image, '/fusion/debug_image', 10)

    def camera_info_callback(self, msg):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]

    def image_callback(self, msg):
        self.image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def detections_callback(self, msg):
        self.detections = msg.detections

    def pointcloud_callback(self, msg):
        if self.image is None or self.detections is None or self.fx is None:
            return

        try:
            transform = self.tf_buffer.lookup_transform(
                'camera_0_color_optical_frame', msg.header.frame_id, rclpy.time.Time())
            cloud_transformed = tf2_sensor_msgs.do_transform_cloud(msg, transform)
        except Exception as e:
            self.get_logger().warn(f"TF transform failed: {e}")
            return

        img = self.image.copy()
        points_3d = []

        for pt in pc2.read_points(cloud_transformed, field_names=["x", "y", "z"], skip_nans=True):
            x, y, z = pt
            if z <= 0 or x <= 0.05 or x > 6.0:
                continue
            u = int((x * self.fx / z) + self.cx)
            v = int((y * self.fy / z) + self.cy)
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                points_3d.append((u, v, x, y, z))

        cmap = plt.cm.get_cmap("hsv", 256)
        cmap = np.array([cmap(i)[:3] for i in range(256)]) * 255

        for det in self.detections:
            w = int(det.bbox.size.x)
            h = int(det.bbox.size.y)
            cx = int(det.bbox.center.position.x)
            cy = int(det.bbox.center.position.y)

            box = [cx / img.shape[1], cy / img.shape[0], w / img.shape[1], h / img.shape[0]]
            x_vals, y_vals, z_vals = [], [], []

            for u, v, x, y, z in points_3d:
                if rectContains(box, (u, v), img.shape[1], img.shape[0], shrink_factor=0.3):
                    x_vals.append(x)
                    y_vals.append(y)
                    z_vals.append(z)
                    color = cmap[int(min(255, 510.0 / max(x, 0.1))) % 256]
                    cv2.circle(img, (u, v), 2, tuple(map(int, color)), -1)

            if x_vals:
                dist_x = get_best_distance(filter_outliers(x_vals))
                dist_y = get_best_distance(filter_outliers(y_vals))
                dist_z = get_best_distance(filter_outliers(z_vals))

                if dist_x is not None and dist_y is not None and dist_z is not None:
                    label_text = f"X:{dist_x:.2f} Y:{dist_y:.2f} Z:{dist_z:.2f} m"
                    cv2.putText(img, label_text, (cx - w//2, cy - h//2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                    cv2.rectangle(img, (cx - w//2, cy - h//2), (cx + w//2, cy + h//2),
                                  (255, 0, 0), 2)
                    self.get_logger().info(
                        f"Detection at ({cx},{cy}) | X: {dist_x:.2f} Y: {dist_y:.2f} Z: {dist_z:.2f} m")
                else:
                    self.get_logger().warn(f"No valid distances inside box at ({cx},{cy})")
            else:
                self.get_logger().warn(f"No valid 3D points in bounding box at ({cx},{cy})")

        self.publisher_debug.publish(self.bridge.cv2_to_imgmsg(img, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraFusionNode()
    rclpy.spin(node)
    rclpy.shutdown()

