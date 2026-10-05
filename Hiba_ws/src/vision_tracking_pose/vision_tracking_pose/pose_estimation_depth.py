#!/usr/bin/env python3
import cv2
import numpy as np
from typing import List, Tuple

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSDurabilityPolicy, QoSReliabilityPolicy
from rclpy.duration import Duration

import message_filters
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import CameraInfo, Image
from geometry_msgs.msg import TransformStamped
from vision_tracking_pose_msgs.msg import Detection, DetectionArray, KeyPoint3D, KeyPoint3DArray, BoundingBox3D


class Detect3DNode(Node):
    def __init__(self):
        super().__init__("detect_3d_node")

        # Parameters
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("maximum_detection_threshold", 0.3)
        self.declare_parameter("depth_image_units_divisor", 1.0)
        self.declare_parameter("depth_topic", "/a200_0846/sensors/camera_0/depth/image")
        self.declare_parameter("depth_info_topic", "/a200_0846/sensors/camera_0/depth/camera_info")
        self.declare_parameter("detection_topic", "/detections")

        # Get values
        self.target_frame = self.get_parameter("target_frame").get_parameter_value().string_value
        self.max_threshold = self.get_parameter("maximum_detection_threshold").get_parameter_value().double_value
        self.depth_divisor = self.get_parameter("depth_image_units_divisor").get_parameter_value().double_value
        self.depth_topic = self.get_parameter("depth_topic").get_parameter_value().string_value
        self.depth_info_topic = self.get_parameter("depth_info_topic").get_parameter_value().string_value
        self.detection_topic = self.get_parameter("detection_topic").get_parameter_value().string_value

        # TF and bridge
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.bridge = CvBridge()

        # Publisher
        self.pub_3d = self.create_publisher(DetectionArray, "detections_3d", 10)

        # Subscribers with sync
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1
        )
        self.sub_depth = message_filters.Subscriber(self, Image, self.depth_topic, qos_profile=qos_profile)
        self.sub_info = message_filters.Subscriber(self, CameraInfo, self.depth_info_topic, qos_profile=qos_profile)
        self.sub_dets = message_filters.Subscriber(self, DetectionArray, self.detection_topic)

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [self.sub_depth, self.sub_info, self.sub_dets], queue_size=10, slop=0.5
        )
        self.sync.registerCallback(self.on_detections)

    def on_detections(self, depth_msg, cam_info_msg, dets_msg):
        detections_out = DetectionArray()
        detections_out.header = dets_msg.header
        detections_out.header.frame_id = self.target_frame
        detections_out.detections = self.process_detections(depth_msg, cam_info_msg, dets_msg)
        self.pub_3d.publish(detections_out)

    def process_detections(self, depth_msg, cam_info, dets_msg):
        if not dets_msg.detections:
            return []

        tf = self.get_transform(cam_info.header.frame_id, dets_msg.header.stamp)
        if tf is None:
            return []

        depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
        detections = []

        for det in dets_msg.detections:
            bbox3d = self.convert_bb_to_3d(depth, cam_info, det)
            if bbox3d is None:
                continue

            bbox3d = self.transform_3d_box(bbox3d, tf[0], tf[1])
            bbox3d.frame_id = self.target_frame

            # SANITY GATE: drop physically impossible detections at the source.
            # When the bbox center / hip pixel lands on BACKGROUND instead of
            # the person (partial occlusion, range edges, thin pose), the
            # sampled depth belongs to a wall — the projected "person" ends up
            # meters underground/in the air or far outside sensor range. Those
            # junk positions were 35% of published detections and poisoned
            # everything downstream (velocity spikes, carrot placement, solver
            # failures). A real person's torso center in base_link sits within
            # a narrow height band and inside the camera's usable range.
            gz = bbox3d.center.position.z
            gd = float(np.hypot(bbox3d.center.position.x, bbox3d.center.position.y))
            if not (-0.4 <= gz <= 2.2) or gd > 8.0 or not np.isfinite(gz):
                continue  # background-depth artifact, not a person

            det.bbox3d = bbox3d

            if det.keypoints.data:
                kps3d = self.convert_keypoints_to_3d(depth, cam_info, det)
                kps3d = self.transform_3d_keypoints(kps3d, tf[0], tf[1])
                kps3d.frame_id = self.target_frame
                det.keypoints3d = kps3d

            detections.append(det)
        return detections

    def convert_bb_to_3d(self, depth_img, cam_info, det):
        cx, cy = int(det.bbox.center.position.x), int(det.bbox.center.position.y)
        sx, sy = int(det.bbox.size.x), int(det.bbox.size.y)

        u_min, u_max = max(cx - sx // 2, 0), min(cx + sx // 2, depth_img.shape[1] - 1)
        v_min, v_max = max(cy - sy // 2, 0), min(cy + sy // 2, depth_img.shape[0] - 1)
        roi = depth_img[v_min:v_max, u_min:u_max].astype(np.float32) / self.depth_divisor
        roi = np.where(np.isfinite(roi), roi, 0)
        valid_roi = roi[roi > 0]

        if valid_roi.size == 0:
            return None

        z_center = depth_img[cy, cx].astype(np.float32) / self.depth_divisor
        if not np.isfinite(z_center) or z_center <= 0:
            return None

        mask_z = np.abs(valid_roi - z_center) <= self.max_threshold
        filtered = valid_roi[mask_z]
        if filtered.size == 0:
            return None

        z_min, z_max = np.min(filtered), np.max(filtered)
        z = (z_min + z_max) / 2
        if z <= 0:
            return None

        k = cam_info.k
        fx, fy, px, py = k[0], k[4], k[2], k[5]
        x = z * (cx - px) / fx
        y = z * (cy - py) / fy
        self.get_logger().info(f"[DEBUG] cx={cx} cy={cy} z_center={z_center:.3f} -> x={x:.3f} y={y:.3f} z={z:.3f}")
        w = z * (sx / fx)
        h = z * (sy / fy)

        bbox = BoundingBox3D()
        bbox.center.position.x = float(x)
        bbox.center.position.y = float(y)
        bbox.center.position.z = float(z)
        bbox.size.x = float(w)
        bbox.size.y = float(h)
        bbox.size.z = float(z_max - z_min)
        return bbox

    def convert_keypoints_to_3d(self, depth_img, cam_info, det):
        kps = np.array([[kp.point.x, kp.point.y] for kp in det.keypoints.data], dtype=np.int32)
        u = np.clip(kps[:, 1], 0, cam_info.height - 1)
        v = np.clip(kps[:, 0], 0, cam_info.width - 1)
        z = depth_img[u, v].astype(np.float32) / self.depth_divisor
        z = np.where(np.isfinite(z), z, 0)

        k = cam_info.k
        fx, fy, px, py = k[0], k[4], k[2], k[5]
        x = z * (v - px) / fx
        y = z * (u - py) / fy

        kps3d = KeyPoint3DArray()
        for i, (x_, y_, z_, d) in enumerate(zip(x, y, z, det.keypoints.data)):
            if float(z_) > 0 and np.isfinite(z_):
                kp3d = KeyPoint3D()
                kp3d.point.x = float(x_)
                kp3d.point.y = float(y_)
                kp3d.point.z = float(z_)
                kp3d.id = int(d.id)
                kp3d.score = float(d.score)
                kps3d.data.append(kp3d)

        return kps3d

    def get_transform(self, frame_id, stamp):
        try:
            tf: TransformStamped = self.tf_buffer.lookup_transform(
                self.target_frame,
                frame_id,
                stamp,
                timeout=Duration(seconds=0.5)
            )
            t = np.array([tf.transform.translation.x,
                          tf.transform.translation.y,
                          tf.transform.translation.z])
            r = np.array([tf.transform.rotation.w,
                          tf.transform.rotation.x,
                          tf.transform.rotation.y,
                          tf.transform.rotation.z])
            return t, r
        except TransformException as ex:
            self.get_logger().error(f"TF lookup failed: {ex}")
            return None

    @staticmethod
    def transform_3d_box(box, t, q):
        p = Detect3DNode.qv_mult(q, np.array([
            box.center.position.x,
            box.center.position.y,
            box.center.position.z
        ])) + t
        box.center.position.x, box.center.position.y, box.center.position.z = p
        s = Detect3DNode.qv_mult(q, np.array([box.size.x, box.size.y, box.size.z]))
        box.size.x, box.size.y, box.size.z = abs(s[0]), abs(s[1]), abs(s[2])
        return box

    @staticmethod
    def transform_3d_keypoints(kps, t, q):
        for kp in kps.data:
            p = Detect3DNode.qv_mult(q, np.array([
                kp.point.x,
                kp.point.y,
                kp.point.z
            ])) + t
            kp.point.x, kp.point.y, kp.point.z = p
        return kps

    @staticmethod
    def qv_mult(q, v):
        q = np.array(q, dtype=np.float64)
        v = np.array(v, dtype=np.float64)
        qvec = q[1:]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        return v + 2 * (q[0] * uv + uuv)


def main(args=None):
    rclpy.init(args=args)
    node = Detect3DNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
