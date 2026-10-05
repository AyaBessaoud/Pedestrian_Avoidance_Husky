#!/usr/bin/env python3

import rclpy
import cv2
import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CameraInfo
from vision_tracking_pose_msgs.msg import DetectionArray, KeyPoint3D, KeyPoint3DArray
from geometry_msgs.msg import Point, TransformStamped
from tf2_ros import TransformException, Buffer, TransformListener
from message_filters import Subscriber, ApproximateTimeSynchronizer
import pyrealsense2 as rs

class DistanceNode(Node):
    def __init__(self):
        super().__init__('distance_node')

        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("maximum_depth", 10.0)
        self.target_frame = self.get_parameter("target_frame").value
        self.max_depth = self.get_parameter("maximum_depth").value

        self.bridge = CvBridge()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.rgb_sub = Subscriber(self, Image, '/a200_0846/sensors/camera_0/color/image', qos_profile=qos_profile_sensor_data)
        self.depth_sub = Subscriber(self, Image, '/a200_0846/sensors/camera_0/depth/image', qos_profile=qos_profile_sensor_data)
        self.info_sub = Subscriber(self, CameraInfo, '/a200_0846/sensors/camera_0/depth/camera_info', qos_profile=qos_profile_sensor_data)
        self.detection_sub = Subscriber(self, DetectionArray, '/tracking', qos_profile=qos_profile_sensor_data)

        self.ts = ApproximateTimeSynchronizer(
            [self.rgb_sub, self.depth_sub, self.info_sub, self.detection_sub],
            queue_size=10,
            slop=0.5
        )
        self.ts.registerCallback(self.sync_callback)

        self.publisher = self.create_publisher(DetectionArray, '/detections_3d', 10)

    def sync_callback(self, rgb_msg, depth_msg, info_msg, detections_msg):
        depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='passthrough')
        intr = rs.intrinsics()
        intr.width, intr.height = info_msg.width, info_msg.height
        intr.ppx, intr.ppy = info_msg.k[2], info_msg.k[5]
        intr.fx, intr.fy = info_msg.k[0], info_msg.k[4]
        intr.model, intr.coeffs = rs.distortion.none, [0, 0, 0, 0, 0]

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                info_msg.header.frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2)
            )
        except TransformException as ex:
            self.get_logger().warn(f'TF transform failed: {str(ex)}')
            return

        updated_msg = DetectionArray()
        updated_msg.header = detections_msg.header
        updated_msg.header.frame_id = self.target_frame

        for det in detections_msg.detections:
            keypoints3d = KeyPoint3DArray()
            keypoints3d.frame_id = self.target_frame
            depth_distance = None

            # Try to use keypoint ID 13 (e.g. right hip)
            for kp in det.keypoints.data:
                if kp.id == 13:
                    u, v = int(kp.point.x), int(kp.point.y)
                    if 0 <= u < depth_image.shape[1] and 0 <= v < depth_image.shape[0]:
                        depth = depth_image[v, u]
                        if depth != 0.0 and not np.isnan(depth) and depth <= self.max_depth:
                            try:
                                x, y, z = rs.rs2_deproject_pixel_to_point(intr, [u, v], float(depth))
                                point = Point(x=x, y=y, z=z)
                                transformed = self.transform_point(point, transform)
                                depth_distance = np.linalg.norm([transformed.x, transformed.y, transformed.z])

                                keypoint3d = KeyPoint3D()
                                keypoint3d.id = kp.id
                                keypoint3d.point = transformed
                                keypoint3d.score = kp.score
                                keypoints3d.data.append(keypoint3d)
                                det.keypoints3d = keypoints3d
                                det.distance = depth_distance  
                            except Exception as e:
                                self.get_logger().warn(f"Transform error (depth): {e}")
                    break

            if det.distance:
                updated_msg.detections.append(det)

        self.publisher.publish(updated_msg)

    def transform_point(self, point: Point, transform: TransformStamped) -> Point:
        t = transform.transform.translation
        q = transform.transform.rotation
        px = np.array([point.x, point.y, point.z])
        qvec = np.array([q.x, q.y, q.z])
        qw = q.w
        uv = np.cross(qvec, px)
        uuv = np.cross(qvec, uv)
        rotated = px + 2 * (qw * uv + uuv)
        return Point(x=rotated[0] + t.x, y=rotated[1] + t.y, z=rotated[2] + t.z)

def main(args=None):
    rclpy.init(args=args)
    node = DistanceNode()
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
