#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import yaml
import numpy as np
from datetime import datetime

from ros2_camera_lidar_fusion.read_yaml import extract_configuration

class CameraCalibrationNode(Node):
    def __init__(self):
        super().__init__('camera_calibration_node')

        config_file = extract_configuration()
        if config_file is None:
            self.get_logger().error("Failed to extract configuration file.")
            return
        
        self.chessboard_rows = config_file['chessboard']['pattern_size']['rows']
        self.chessboard_cols = config_file['chessboard']['pattern_size']['columns']
        self.square_size = config_file['chessboard']['square_size_meters']

        self.image_topic = config_file['camera']['image_topic']
        self.image_width = config_file['camera']['image_size']['width']
        self.image_height = config_file['camera']['image_size']['height']

        self.output_path = config_file['general']['config_folder']
        self.file = config_file['general']['camera_intrinsic_calibration']

        self.image_sub = self.create_subscription(Image, self.image_topic, self.image_callback, 10)
        self.bridge = CvBridge()

        self.obj_points = []
        self.img_points = []

        self.objp = np.zeros((self.chessboard_rows * self.chessboard_cols, 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_cols, 0:self.chessboard_rows].T.reshape(-1, 2)
        self.objp *= self.square_size

        self.get_logger().info("Camera calibration node initialized. Waiting for images...")

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

            ret, corners = cv2.findChessboardCorners(gray, (self.chessboard_cols, self.chessboard_rows), None)

            if ret:
                self.obj_points.append(self.objp)
                refined_corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                )
                self.img_points.append(refined_corners)

                cv2.drawChessboardCorners(cv_image, (self.chessboard_cols, self.chessboard_rows), refined_corners, ret)
                self.get_logger().info("Chessboard detected and points added.")
            else:
                self.get_logger().warn("Chessboard not detected in image.")

            cv2.imshow("Image", cv_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f"Failed to process image: {e}")

    def save_calibration(self):
        if len(self.obj_points) < 10:
            self.get_logger().error("Not enough images for calibration. At least 10 are required.")
            return

        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points, self.img_points, (self.image_width, self.image_height), None, None
        )

        calibration_data = {
            'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'camera_matrix': {
                'rows': 3,
                'columns': 3,
                'data': camera_matrix.tolist()
            },
            'distortion_coefficients': {
                'rows': 1,
                'columns': len(dist_coeffs[0]),
                'data': dist_coeffs[0].tolist()
            },
            'chessboard': {
                'pattern_size': {
                    'rows': self.chessboard_rows,
                    'columns': self.chessboard_cols
                },
                'square_size_meters': self.square_size
            },
            'image_size': {
                'width': 640,
                'height': 480
            },
            'rms_reprojection_error': ret
        }

        output_file = f"{self.output_path}/{self.file}"
        try:
            with open(output_file, 'w') as file:
                yaml.dump(calibration_data, file)
            self.get_logger().info(f"Calibration saved to {output_file}")
        except Exception as e:
            self.get_logger().error(f"Failed to save calibration: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = CameraCalibrationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_calibration()
        node.get_logger().info("Calibration process completed.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
