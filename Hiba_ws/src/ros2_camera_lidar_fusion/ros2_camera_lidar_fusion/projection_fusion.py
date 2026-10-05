#!/usr/bin/env python3

import os
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSDurabilityPolicy

import cv2
import numpy as np
import yaml

from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
from yolo_msgs.msg import DetectionArray  

from ros2_camera_lidar_fusion.read_yaml import extract_configuration


def load_extrinsic_matrix(yaml_path: str) -> np.ndarray:
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"No extrinsic file found: {yaml_path}")
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    if 'extrinsic_matrix' not in data:
        raise KeyError("Missing 'extrinsic_matrix' key in YAML.")
    T = np.array(data['extrinsic_matrix'], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("Extrinsic matrix must be 4x4.")
    return T


def load_camera_calibration(yaml_path: str):
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"No camera calibration file: {yaml_path}")
    with open(yaml_path, 'r') as f:
        calib_data = yaml.safe_load(f)
    camera_matrix = np.array(calib_data['camera_matrix']['data'], dtype=np.float64).reshape(3, 3)
    dist_coeffs = np.array(calib_data['distortion_coefficients']['data'], dtype=np.float64).reshape((1, -1))
    return camera_matrix, dist_coeffs


def pointcloud2_to_xyz_array(cloud_msg: PointCloud2, skip: int = 1) -> np.ndarray:
    field_names = [f.name for f in cloud_msg.fields]
    if not all(k in field_names for k in ('x', 'y', 'z')):
        return np.zeros((0, 3), dtype=np.float32)
    dtype = np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('_', 'V{}'.format(cloud_msg.point_step - 12))
    ])
    raw = np.frombuffer(cloud_msg.data, dtype=dtype)
    points = np.stack((raw['x'], raw['y'], raw['z']), axis=-1)
    return points[::skip] if skip > 1 else points


class LidarCameraProjectionNode(Node):
    def __init__(self):
        super().__init__('lidar_camera_projection_node')

        config = extract_configuration()
        if config is None:
            self.get_logger().error(" Configuration file missing")
            return

        folder = config['general']['config_folder']
        self.T_lidar_to_cam = load_extrinsic_matrix(os.path.join(folder, config['general']['camera_extrinsic_calibration']))
        self.camera_matrix, self.dist_coeffs = load_camera_calibration(os.path.join(folder, config['general']['camera_intrinsic_calibration']))

        lidar_topic = config['lidar']['lidar_topic']
        image_topic = config['camera']['image_topic']
        projected_topic = config['camera']['projected_topic']

        self.get_logger().info(f"Subscribed to:\n- LiDAR: {lidar_topic}\n- Camera: {image_topic}")
        self.get_logger().info(f"Extrinsics:\n{self.T_lidar_to_cam}")
        self.get_logger().info(f"Intrinsics:\n{self.camera_matrix}\nDistortion: {self.dist_coeffs}")

        self.bridge = CvBridge()
        self.skip_rate = 1
        self.pub_image = self.create_publisher(Image, projected_topic, 1)

        # QoS
        qos_lidar = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.BEST_EFFORT, durability=QoSDurabilityPolicy.VOLATILE)
        qos_camera = QoSProfile(depth=10, reliability=QoSReliabilityPolicy.RELIABLE, durability=QoSDurabilityPolicy.VOLATILE)

        # Subscribers
        self.latest_image = None
        self.latest_lidar = None
        self.latest_detections = None

        self.image_sub = self.create_subscription(Image, image_topic, self.image_callback, qos_camera)
        self.lidar_sub = self.create_subscription(PointCloud2, lidar_topic, self.lidar_callback, qos_lidar)
        self.detection_sub = self.create_subscription(DetectionArray, 'tracking', self.detection_callback, 10)

        self.timer = self.create_timer(0.2, self.timer_callback)

    def image_callback(self, msg):
        self.latest_image = msg

    def lidar_callback(self, msg):
        self.latest_lidar = msg

    def detection_callback(self, msg):
        self.latest_detections = msg

    def timer_callback(self):
        if self.latest_image and self.latest_lidar and self.latest_detections:
            self.sync_callback(self.latest_image, self.latest_lidar, self.latest_detections)

    def sync_callback(self, image_msg: Image, lidar_msg: PointCloud2, detections_msg: DetectionArray):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            return

        xyz = pointcloud2_to_xyz_array(lidar_msg, skip=self.skip_rate)
        if xyz.shape[0] == 0:
            self.get_logger().warn("No LiDAR points to project.")
            return

        xyz_h = np.hstack((xyz.astype(np.float64), np.ones((xyz.shape[0], 1))))
        xyz_cam = (self.T_lidar_to_cam @ xyz_h.T).T[:, :3]
        mask = xyz_cam[:, 2] > 0
        xyz_cam = xyz_cam[mask]

        if xyz_cam.shape[0] == 0:
            self.get_logger().info("All points behind camera.")
            return

        try:
            image_points, _ = cv2.projectPoints(
                xyz_cam, np.zeros((3, 1)), np.zeros((3, 1)),
                self.camera_matrix, self.dist_coeffs
            )
        except cv2.error as e:
            self.get_logger().error(f"Projection error: {e}")
            return

        image_points = image_points.reshape(-1, 2)
        h, w = cv_image.shape[:2]
        for u, v in image_points:
            u_int, v_int = int(round(u)), int(round(v))
            if 0 <= u_int < w and 0 <= v_int < h:
                cv2.circle(cv_image, (u_int, v_int), 2, (0, 255, 0), -1)

        self.get_logger().info(f"Projected {len(image_points)} points.")

        # Compute object distances from LiDAR points
        depths = xyz_cam[:, 2]
        for det in detections_msg.detections:
            cx = det.bbox.center.position.x
            cy = det.bbox.center.position.y
            bw = det.bbox.size.x
            bh = det.bbox.size.y
            x_min, x_max = cx - bw / 2, cx + bw / 2
            y_min, y_max = cy - bh / 2, cy + bh / 2

            depth_in_box = []
            for (u, v), d in zip(image_points, depths):
                if x_min <= u <= x_max and y_min <= v <= y_max:
                    depth_in_box.append(d)

            if len(depth_in_box) > 0:
                estimated_depth = float(np.median(depth_in_box))
                cv2.putText(cv_image, f"{estimated_depth:.2f}m", (int(cx), int(cy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                self.get_logger().info(f"[ID {det.id}] Distance = {estimated_depth:.2f} m")
            else:
                self.get_logger().info(f"[ID {det.id}] No LiDAR points in bbox")

        # Publish updated image
        out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        out_msg.header = image_msg.header
        self.pub_image.publish(out_msg)

        cv2.imshow("Projected LiDAR + Distances", cv_image)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = LidarCameraProjectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
