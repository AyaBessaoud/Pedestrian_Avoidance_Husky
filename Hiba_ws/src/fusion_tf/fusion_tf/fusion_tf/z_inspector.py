#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

class ZInspector(Node):
    def __init__(self):
        super().__init__('z_inspector')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/filtered_pointcloud',
            self.listener_callback,
            10)
        self.get_logger().info("Listening to /filtered_pointcloud...")

    def listener_callback(self, msg):
        z_vals = [pt[2] for pt in point_cloud2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True)]
        if not z_vals:
            self.get_logger().warn("No valid Z values.")
            return

        min_z = min(z_vals)
        max_z = max(z_vals)
        avg_z = sum(z_vals) / len(z_vals)

        self.get_logger().info(f"Z range → Min: {min_z:.4f}, Max: {max_z:.4f}, Avg: {avg_z:.4f}")

def main(args=None):
    rclpy.init(args=args)
    node = ZInspector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
