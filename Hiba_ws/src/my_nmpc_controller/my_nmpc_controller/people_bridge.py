#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from people_msgs.msg import People
from std_msgs.msg import Bool

class PeopleBridge(Node):
    def __init__(self):
        super().__init__('people_bridge')
        self.pub = self.create_publisher(Bool, '/pedestrian_detected', 10)
        self.sub = self.create_subscription(
            People, '/people', self.callback, 10)
        self.get_logger().info("PeopleBridge started")

    def callback(self, msg):
        detected = Bool()
        detected.data = len(msg.people) > 0
        self.pub.publish(detected)

def main():
    rclpy.init()
    node = PeopleBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
