import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from tf2_ros.transform_listener import QoSProfile
from tf_transformations import quaternion_matrix
import numpy as np

from rclpy.qos import QoSProfile as RclpyQoSProfile, ReliabilityPolicy, HistoryPolicy

class ExtrinsicMatrixNode(Node):
    def __init__(self):
        super().__init__('extrinsic_matrix_node')

        # Create buffer and listener with remapped TF topics
        qos = RclpyQoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST
        )

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self,
            qos=qos
        )

        # Declare and remap topic names
        self.declare_parameter('tf_topic', '/tf')
        self.declare_parameter('tf_static_topic', '/tf_static')

        self.tf_topic = self.get_parameter('tf_topic').get_parameter_value().string_value
        self.tf_static_topic = self.get_parameter('tf_static_topic').get_parameter_value().string_value

        # Remapping happens externally; here we just use a timer to try looking up the transform
        self.timer = self.create_timer(1.0, self.extract_matrix)

    def extract_matrix(self):
        from_frame = 'rplidar_link'
        to_frame = 'camera_0_link'

        try:
            transform = self.tf_buffer.lookup_transform(
                to_frame,
                from_frame,
                rclpy.time.Time()
            )

            t = transform.transform.translation
            q = transform.transform.rotation

            T = quaternion_matrix([q.x, q.y, q.z, q.w])
            T[0, 3] = t.x
            T[1, 3] = t.y
            T[2, 3] = t.z

            print("\nExtrinsic Matrix (LiDAR → Camera):")
            print(np.array_str(T, precision=4, suppress_small=True))

            self.destroy_node()
        except Exception as e:
            self.get_logger().warn(f"Transform not available yet: {str(e)}")

def main():
    rclpy.init()
    node = ExtrinsicMatrixNode()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
