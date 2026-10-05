#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, PointCloud2
import laser_geometry.laser_geometry as lg

class ScanToCloudNode(Node):
    def __init__(self):
        super().__init__('scan_to_cloud_node')
        self.lp = lg.LaserProjection()

        # Subscribe to the /scan topic
        self.scan_sub = self.create_subscription(
            LaserScan,
            #'/a200_0846/rear/scan', 
            '/a200_0846/sensors/lidar2d_0/scan',# Change this if your topic is namespaced
            self.scan_callback,
            10
        )

        # Publisher for PointCloud2
        self.pc_pub = self.create_publisher(PointCloud2, '/cloud', 10)

        self.first_message = True
        self.get_logger().info("ScanToCloudNode started successfully!")

    def scan_callback(self, msg: LaserScan):
        try:
            pc2_msg = self.lp.projectLaser(msg)
            pc2_msg.header.frame_id = msg.header.frame_id
            pc2_msg.header.stamp = msg.header.stamp
            self.pc_pub.publish(pc2_msg)

            if self.first_message:
                self.get_logger().info("Publishing PointCloud2 from LaserScan...")
                self.first_message = False
        except Exception as e:
            self.get_logger().error(f"Error converting LaserScan to PointCloud2: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ScanToCloudNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
