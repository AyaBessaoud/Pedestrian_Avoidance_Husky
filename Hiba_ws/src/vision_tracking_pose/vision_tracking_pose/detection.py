#!/usr/bin/env python3
import cv2
import torch
import numpy as np

from cv_bridge import CvBridge
from ultralytics import YOLO
from sensor_msgs.msg import Image
from vision_tracking_pose_msgs.msg import (
    Detection, DetectionArray,
    Point2D, Pose2D, Vector2,
    BoundingBox2D, KeyPoint2D, KeyPoint2DArray
)

import rclpy
from rclpy.node import Node


class Yolo2DNode(Node):
    def __init__(self):
        super().__init__('yolo_2d_node')

        self.declare_parameter('model','/home/hiba/clearpath_ws/src/vision_tracking_pose/vision_tracking_pose/yolov8n-pose.pt') 
        #self.declare_parameter('image_topic', '/camera/camera/color/image_raw')
        self.declare_parameter('image_topic', '/a200_0846/sensors/camera_0/color/image')

        model_path = self.get_parameter('model').get_parameter_value().string_value
        image_topic = self.get_parameter('image_topic').get_parameter_value().string_value

        self.model = YOLO(model_path).to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.fuse()
        self.get_logger().info(f"Model loaded on: {'cuda' if torch.cuda.is_available() else 'cpu'}")

        self.bridge = CvBridge()
        self.sub_rgb = self.create_subscription(Image, image_topic, self.image_cb, 10)
        self.pub_detections = self.create_publisher(DetectionArray, 'detections', 10)

    def image_cb(self, msg):
        img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        results = self.model.predict(img, stream=False, verbose=False)[0]

        msg_array = DetectionArray()
        msg_array.header = msg.header

        for i, det in enumerate(results.boxes):
            det_msg = Detection()
            det_msg.id = str(i)
            det_msg.class_id = int(det.cls.item())
            det_msg.class_name = self.model.names[det_msg.class_id]
            det_msg.score = float(det.conf.item())

            bbox = det.xywh[0]
            bbox_msg = BoundingBox2D()
            bbox_msg.center = Pose2D()
            bbox_msg.center.position = Point2D(x=float(bbox[0]), y=float(bbox[1]))
            bbox_msg.center.theta = 0.0
            bbox_msg.size = Vector2(x=float(bbox[2]), y=float(bbox[3]))
            det_msg.bbox = bbox_msg

            if results.keypoints is not None:
                kps_2d = KeyPoint2DArray()
                for kp_id, (p, conf) in enumerate(zip(results.keypoints.xy[i], results.keypoints.conf[i])):
                    if conf >= 0.5:
                        kp = KeyPoint2D()
                        kp.id = kp_id + 1
                        kp.point.x = float(p[0])
                        kp.point.y = float(p[1])
                        kp.score = float(conf)
                        kps_2d.data.append(kp)
                det_msg.keypoints = kps_2d

            msg_array.detections.append(det_msg)

        self.pub_detections.publish(msg_array)


def main():
    rclpy.init()
    node = Yolo2DNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
