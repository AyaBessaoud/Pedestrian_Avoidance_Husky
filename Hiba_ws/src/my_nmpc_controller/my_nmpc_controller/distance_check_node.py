import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan
import numpy as np


class DistanceCheck(Node):

    def __init__(self):
        super().__init__("distance_check_node")
        scan_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=5)
        self.scan_sub = self.create_subscription(
            LaserScan, "/cpr_a200_0000/sensors/lidar2d_0/scan", self.scan_cb, scan_qos
        )
        self.get_logger().info("=" * 60)
        self.get_logger().info("Distance check node ready.")
        self.get_logger().info("Place robot near walls and watch the output.")
        self.get_logger().info("=" * 60)

    def scan_cb(self, msg):
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        valid = (ranges > 0.05) & (ranges < 20.0) & np.isfinite(ranges)
        if not np.any(valid):
            self.get_logger().warn("No valid scan readings!")
            return
        r = ranges[valid]
        a = angles[valid]
        lx = r * np.cos(a)
        ly = r * np.sin(a)

        def sector(a_min_deg, a_max_deg):
            mask = (a >= np.radians(a_min_deg)) & (a <= np.radians(a_max_deg))
            if np.any(mask):
                return float(np.min(r[mask]))
            return float("inf")

        front = sector(-30, 30)
        left = sector(30, 150)
        rear_pos = sector(150, 180)
        rear_neg = sector(-180, -150)
        rear = min(rear_pos, rear_neg)
        right = sector(-150, -30)
        d_min = float(np.min(r))
        idx_min = np.argmin(r)
        a_min = float(a[idx_min])
        self.get_logger().info(
            f"\n  ┌─────────────────────────────────────────┐\n  │  FRONT  ({-30:+.0f}° to {30:+.0f}°):  {front:6.3f} m         │\n  │  LEFT   ({30:+.0f}° to {150:+.0f}°):  {left:6.3f} m         │\n  │  RIGHT  ({-150:+.0f}° to {-30:+.0f}°):  {right:6.3f} m         │\n  │  REAR   ({150:+.0f}°/{-150:+.0f}°):  {rear:6.3f} m         │\n  ├─────────────────────────────────────────┤\n  │  CLOSEST: {d_min:.3f} m at {np.degrees(a_min):+.1f}°            │\n  └─────────────────────────────────────────┘",
            throttle_duration_sec=1.0,
        )
        if d_min < 0.4:
            self.get_logger().warn(
                f"*** VERY CLOSE: {d_min:.3f} m at {np.degrees(a_min):+.1f}° ***"
            )


def main():
    rclpy.init()
    node = DistanceCheck()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
