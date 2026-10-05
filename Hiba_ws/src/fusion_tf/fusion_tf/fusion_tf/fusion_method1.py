#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, CameraInfo, Image
from yolo_msgs.msg import DetectionArray, DetectedPerson
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge
import numpy as np
import math
import cv2

class LidarScanFusionNode(Node):
    def __init__(self):
        super().__init__('lidar_scan_fusion_node')

        # Parameters
        self.declare_parameter("camera_frame", "camera_0_color_optical_frame")
        self.declare_parameter("lidar_frame", "lidar2d_0_laser")
        self.camera_frame = self.get_parameter("camera_frame").value
        self.lidar_frame = self.get_parameter("lidar_frame").value

        # Camera intrinsics
        self.fx = self.fy = self.cx = self.cy = None
        self.camera_model_ready = False
        self.detections = None
        self.latest_image = None
        self.bridge = CvBridge()

        # TF2 setup
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers
        self.create_subscription(CameraInfo, "/a200_0846/sensors/camera_0/color/camera_info", self.camera_info_callback, 10)
        self.create_subscription(Image, "/a200_0846/sensors/camera_0/color/image", self.image_callback, 10)
        self.create_subscription(DetectionArray, "/tracking", self.detections_callback, 10)
        self.create_subscription(LaserScan, "/a200_0846/sensors/lidar2d_0/scan", self.scan_callback, 10)

        # Publishers
        self.person_pub = self.create_publisher(DetectedPerson, "/detected_person_scan", 10)
        self.debug_pub = self.create_publisher(Image, "/fusion/debug_image", 10)

    def camera_info_callback(self, msg):
        if not self.camera_model_ready:
            self.fx = msg.k[0]
            self.fy = msg.k[4]
            self.cx = msg.k[2]
            self.cy = msg.k[5]
            self.camera_model_ready = True
            self.get_logger().info("✅ Camera intrinsics received.")

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Image conversion error: {e}")

    def detections_callback(self, msg):
        self.detections = msg

    def shrink_bbox(self, det, shrink_factor=0.1):
        """Shrink bounding box by a percentage."""
        cx = det.bbox.center.position.x
        cy = det.bbox.center.position.y
        w = det.bbox.size.x * (1 - shrink_factor)
        h = det.bbox.size.y * (1 - shrink_factor)
        xmin = int(cx - w / 2)
        xmax = int(cx + w / 2)
        ymin = int(cy - h / 2)
        ymax = int(cy + h / 2)
        return xmin, xmax, ymin, ymax

    def scan_callback(self, scan_msg):
        if not self.camera_model_ready or self.detections is None or self.latest_image is None:
            return

        try:
            tf = self.tf_buffer.lookup_transform(
                self.camera_frame,
                scan_msg.header.frame_id,
                rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return

        image = self.latest_image.copy()
        angle = scan_msg.angle_min
        angle_inc = scan_msg.angle_increment

        for i, r in enumerate(scan_msg.ranges):
            if r < scan_msg.range_min or r > scan_msg.range_max:
                angle += angle_inc
                continue

            # LiDAR point in LiDAR frame
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            z = 0.0

            pt = PointStamped()
            pt.header = scan_msg.header
            pt.point.x = x
            pt.point.y = y
            pt.point.z = z

            try:
                pt_cam = tf2_geometry_msgs.do_transform_point(pt, tf)
            except Exception as e:
                self.get_logger().warn(f"TF transform failed: {e}")
                angle += angle_inc
                continue

            # Project to image
            X = pt_cam.point.x
            Y = pt_cam.point.y
            Z = pt_cam.point.z
            if Z <= 0:
                angle += angle_inc
                continue

            u = int(self.fx * X / Z + self.cx)
            v = int(self.fy * Y / Z + self.cy)

            matched = False
            for det in self.detections.detections:
                xmin, xmax, ymin, ymax = self.shrink_bbox(det, shrink_factor=0.5)

                if xmin <= u <= xmax and ymin <= v <= ymax:
                    matched = True
                    #distance = math.sqrt(X**2 + Y**2 + Z**2)  # More precise than 2D LiDAR range
                    distance = z
                    msg = DetectedPerson()
                    msg.id = int(det.id)
                    msg.distance = float(distance)
                    msg.angle = float(angle)
                    self.person_pub.publish(msg)
                    self.get_logger().info(f"✅ Matched ID {msg.id} → {msg.distance:.2f} m at {msg.angle:.2f} rad")
                    break

            if 0 <= u < image.shape[1] and 0 <= v < image.shape[0]:
                color = (0, 255, 0) if matched else (0, 0, 255)
                cv2.circle(image, (u, v), 2, color, -1)

            angle += angle_inc

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
            debug_msg.header.stamp = self.get_clock().now().to_msg()
            self.debug_pub.publish(debug_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish debug image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = LidarScanFusionNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == "__main__":
    main()

