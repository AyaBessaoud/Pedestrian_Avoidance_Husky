#!/usr/bin/env python3

import cv2
import random
import numpy as np
from typing import Tuple

import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from cv_bridge import CvBridge
from ultralytics.utils.plotting import Annotator, colors

from sensor_msgs.msg import Image
from visualization_msgs.msg import Marker, MarkerArray
from vision_tracking_pose_msgs.msg  import DetectionArray, KeyPoint3D

class DebugNode(Node):
    def __init__(self):
        super().__init__('debug_node')

        self._class_to_color = {}
        self.bridge = CvBridge()

        # Parameters
        self.declare_parameter("image_topic", "a200_0846/sensors/camera_0/color/image")  
        self.declare_parameter("detections_topic", "/detections_3d")

        # Get parameter values
        image_topic = self.get_parameter("image_topic").value
        detections_topic = self.get_parameter("detections_topic").value

        # Subscriptions
        self.image_sub = self.create_subscription(Image, image_topic, self.image_cb, 10)
        self.detections_sub = self.create_subscription(DetectionArray, detections_topic, self.detections_cb, 10)

        # Publishers
        self.debug_img_pub = self.create_publisher(Image, "/dbg_image", 10)
        self.bb_marker_pub = self.create_publisher(MarkerArray, "/dbg_bb_markers", 10)
        self.kp_marker_pub = self.create_publisher(MarkerArray, "/dbg_kp_markers", 10)
        self.id_marker_pub = self.create_publisher(MarkerArray, "/dbg_id_markers", 10)

        self.last_image_msg = None

    def image_cb(self, msg: Image):
        self.last_image_msg = msg

    def detections_cb(self, msg: DetectionArray):
        if self.last_image_msg is None:
            return

        image = self.bridge.imgmsg_to_cv2(self.last_image_msg, desired_encoding='bgr8')
        bb_markers = MarkerArray()
        kp_markers = MarkerArray()
        id_markers = MarkerArray()

        for det in msg.detections:
            class_name = det.class_name
            if class_name not in self._class_to_color:
                self._class_to_color[class_name] = tuple(random.randint(0, 255) for _ in range(3))
            color = self._class_to_color[class_name]

            # Draw 2D box
            image = self.draw_box(image, det, color)
            # Draw 2D keypoints
            image = self.draw_keypoints(image, det)

            # Draw 3D bounding box
            if det.bbox3d.frame_id:
                marker = self.create_bb_marker(det, color)
                marker.header.stamp = self.last_image_msg.header.stamp
                marker.id = len(bb_markers.markers)
                bb_markers.markers.append(marker)

                # Add ID marker
                id_marker = self.create_id_marker(det)
                id_marker.header.stamp = self.last_image_msg.header.stamp
                id_marker.id = len(id_markers.markers)
                id_markers.markers.append(id_marker)

            # Draw 3D keypoints
            if det.keypoints3d.frame_id:
                for kp in det.keypoints3d.data:
                    marker = self.create_kp_marker(kp)
                    marker.header.frame_id = det.keypoints3d.frame_id
                    marker.header.stamp = self.last_image_msg.header.stamp
                    marker.id = len(kp_markers.markers)
                    kp_markers.markers.append(marker)

        # Publish outputs
        dbg_msg = self.bridge.cv2_to_imgmsg(image, encoding='bgr8')
        dbg_msg.header = self.last_image_msg.header
        self.debug_img_pub.publish(dbg_msg)
        self.bb_marker_pub.publish(bb_markers)
        self.kp_marker_pub.publish(kp_markers)
        self.id_marker_pub.publish(id_markers)

    def draw_box(self, image: np.ndarray, detection, color: Tuple[int]) -> np.ndarray:
        x, y = detection.bbox.center.position.x, detection.bbox.center.position.y
        w, h = detection.bbox.size.x, detection.bbox.size.y

        top_left = (int(x - w / 2), int(y - h / 2))
        bottom_right = (int(x + w / 2), int(y + h / 2))
        label = f"{detection.class_name} ({detection.id}) ({detection.score:.2f})"

        cv2.rectangle(image, top_left, bottom_right, color, 2)
        cv2.putText(image, label, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return image

    def draw_keypoints(self, image: np.ndarray, detection) -> np.ndarray:
        annotator = Annotator(image)
        for kp in detection.keypoints.data:
            pt = (int(kp.point.x), int(kp.point.y))
            color_k = colors(kp.id - 1)
            cv2.circle(image, pt, 5, color_k, -1)
            cv2.putText(image, str(kp.id), (pt[0]+4, pt[1]-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_k, 1)
        return image

    def create_bb_marker(self, detection, color: Tuple[int]) -> Marker:
        bbox3d = detection.bbox3d
        marker = Marker()
        marker.header.frame_id = bbox3d.frame_id
        marker.type = Marker.CUBE
        marker.action = Marker.ADD
        marker.pose.position = bbox3d.center.position
        marker.pose.orientation.w = 1.0
        marker.scale.x = bbox3d.size.x
        marker.scale.y = bbox3d.size.y
        marker.scale.z = bbox3d.size.z
        marker.color.r = color[0] / 255.0
        marker.color.g = color[1] / 255.0
        marker.color.b = color[2] / 255.0
        marker.color.a = 0.5
        marker.lifetime = Duration(seconds=0.5).to_msg()
        return marker

    def create_kp_marker(self, kp: KeyPoint3D) -> Marker:
        marker = Marker()
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = kp.point
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.05
        marker.color.r = (1.0 - kp.score)
        marker.color.b = kp.score
        marker.color.a = 0.7
        marker.lifetime = Duration(seconds=0.5).to_msg()
        return marker

    def create_id_marker(self, detection) -> Marker:
        marker = Marker()
        marker.header.frame_id = detection.bbox3d.frame_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position = detection.bbox3d.center.position
        marker.pose.position.z += 0.5  # Offset text above the box
        marker.scale.z = 0.2
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.text = str(detection.id)
        marker.lifetime = Duration(seconds=0.5).to_msg()
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = DebugNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
