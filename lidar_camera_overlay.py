#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, Image, CameraInfo
from cv_bridge import CvBridge
import numpy as np
import cv2
from image_geometry import PinholeCameraModel
import tf2_ros
from tf2_geometry_msgs import do_transform_point
from geometry_msgs.msg import PointStamped

SCAN_TOPIC = "/a200_0846/sensors/lidar2d_0/scan"
IMAGE_TOPIC = "/a200_0846/sensors/camera_0/color/image"
CAMERA_INFO_TOPIC = "/a200_0846/sensors/camera_0/color/camera_info"
OUTPUT_TOPIC = "/a200_0846/lidar_camera_overlay"


class LidarCameraOverlay(Node):

    def __init__(self):
        super().__init__("lidar_camera_overlay")
        self.bridge = CvBridge()
        self.cam_model = PinholeCameraModel()
        self.have_cam_info = False
        self.latest_image = None
        self.latest_image_frame = None
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_cb, 10)
        self.create_subscription(Image, IMAGE_TOPIC, self.image_cb, qos_profile_sensor_data)
        self.create_subscription(LaserScan, SCAN_TOPIC, self.scan_cb, qos_profile_sensor_data)
        self.publisher = self.create_publisher(Image, OUTPUT_TOPIC, 10)
        self.get_logger().info(
            f"Waiting for camera_info on {CAMERA_INFO_TOPIC}, image on {IMAGE_TOPIC}, scan on {SCAN_TOPIC}..."
        )

    def camera_info_cb(self, msg: CameraInfo):
        if not self.have_cam_info:
            self.cam_model.fromCameraInfo(msg)
            self.have_cam_info = True
            self.get_logger().info("Camera intrinsics received.")

    def image_cb(self, msg: Image):
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.latest_image_frame = msg.header.frame_id

    def scan_cb(self, msg: LaserScan):
        if not self.have_cam_info or self.latest_image is None:
            return
        img = self.latest_image.copy()
        target_frame = self.latest_image_frame
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame, msg.header.frame_id, rclpy.time.Time()
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=2.0)
            return
        angle = msg.angle_min
        drawn = 0
        for r in msg.ranges:
            if msg.range_min < r < msg.range_max:
                x = r * np.cos(angle)
                y = r * np.sin(angle)
                pt = PointStamped()
                pt.header = msg.header
                pt.point.x = x
                pt.point.y = y
                pt.point.z = 0.0
                pt_cam = do_transform_point(pt, transform)
                if pt_cam.point.z > 0:
                    u, v = self.cam_model.project3dToPixel(
                        (pt_cam.point.x, pt_cam.point.y, pt_cam.point.z)
                    )
                    if 0 <= int(u) < img.shape[1] and 0 <= int(v) < img.shape[0]:
                        cv2.circle(img, (int(u), int(v)), 3, (0, 0, 255), -1)
                        drawn += 1
            angle += msg.angle_increment
        out_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        out_msg.header = msg.header
        self.publisher.publish(out_msg)


def main():
    rclpy.init()
    node = LidarCameraOverlay()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
