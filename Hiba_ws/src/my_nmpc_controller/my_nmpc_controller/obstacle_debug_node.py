import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener
import numpy as np


class ObstacleDebug(Node):

    def __init__(self):
        super().__init__("obstacle_debug_node")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        scan_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=5)
        self.scan_sub = self.create_subscription(
            LaserScan, "/cpr_a200_0000/sensors/lidar2d_0/scan", self.scan_cb, scan_qos
        )
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self.pose_cb, 10
        )
        self.d_safe = 0.5
        self.influence = 1.15
        self.n_sectors = 6
        self.obs_dist_limit = 3.0
        self.obstacles = []
        self.state_now = None
        self.is_localized = False
        self.create_timer(1.0, self.print_debug)
        self.get_logger().info("Obstacle debug node ready — place robot between shelves")

    def pose_cb(self, msg):
        self.is_localized = True
        self.get_logger().info("Localized")

    def scan_cb(self, msg):
        tmp = []
        angle = msg.angle_min
        step = 5
        for i in range(0, len(msg.ranges), step):
            r = msg.ranges[i]
            if 0.15 < r < 6.0:
                tmp.append((r * np.cos(angle), r * np.sin(angle), r, angle))
            angle += msg.angle_increment * step
        tmp.sort(key=lambda o: o[2])
        self.obstacles = tmp

    def get_world_obs(self):
        if self.state_now is None:
            return []
        x, y, th = self.state_now
        c, s = (np.cos(th), np.sin(th))
        result = []
        for lx, ly, dist, ang_local in self.obstacles:
            wx = x + c * lx - s * ly
            wy = y + s * lx + c * ly
            result.append((wx, wy, dist, ang_local))
        return result

    def select_obstacles(self, world_obs):
        if self.state_now is None:
            return {}
        x, y, th = self.state_now
        sector_size = 2 * np.pi / self.n_sectors
        best = {}
        for ox, oy, dist, _ in world_obs:
            if dist > self.obs_dist_limit:
                continue
            angle_world = np.arctan2(oy - y, ox - x)
            angle_robot = np.arctan2(np.sin(angle_world - th), np.cos(angle_world - th))
            sector = int((angle_robot + np.pi) / sector_size) % self.n_sectors
            if sector not in best or dist < best[sector][2]:
                best[sector] = (ox, oy, dist, np.degrees(angle_robot))
        return best

    def angle_to_label(self, deg):
        a = abs(deg)
        if a <= 30:
            return "FRONT"
        elif a <= 60:
            return "FRONT-SIDE"
        elif a <= 120:
            return "SIDE"
        elif a <= 150:
            return "REAR-SIDE"
        else:
            return "REAR"

    def print_debug(self):
        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            q = t.transform.rotation
            th = np.arctan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y**2 + q.z**2))
            self.state_now = [t.transform.translation.x, t.transform.translation.y, th]
        except Exception:
            self.get_logger().warn(
                "Waiting for TF — make sure robot is localized", throttle_duration_sec=3.0
            )
            return
        if not self.obstacles:
            self.get_logger().warn("No scan data yet")
            return
        world_obs = self.get_world_obs()
        selected = self.select_obstacles(world_obs)
        x, y, th = self.state_now
        all_dists = [o[2] for o in self.obstacles]
        closest_raw = min(all_dists)
        closest_ang = np.degrees(self.obstacles[all_dists.index(closest_raw)][3])
        lines = []
        lines.append("=" * 60)
        lines.append(f"ROBOT pos=({x:.2f}, {y:.2f}) heading={np.degrees(th):.1f}°")
        lines.append(
            f"RAW SCAN: {len(self.obstacles)} points | closest={closest_raw:.3f}m at {closest_ang:.1f}° (robot frame)"
        )
        lines.append("")
        lines.append("── RAW LIDAR by sector (closest point each) ──")
        sector_size = 2 * np.pi / self.n_sectors
        sector_data = {}
        for lx, ly, dist, ang_local in self.obstacles:
            if dist > self.obs_dist_limit:
                continue
            sector = int((ang_local + np.pi) / sector_size) % self.n_sectors
            if sector not in sector_data or dist < sector_data[sector][0]:
                sector_data[sector] = (dist, np.degrees(ang_local))
        sector_names = {
            0: "REAR-LEFT  (-180° to -120°)",
            1: "LEFT       (-120° to  -60°)",
            2: "FRONT-LEFT ( -60° to    0°)",
            3: "FRONT-RIGHT(   0° to  +60°)",
            4: "RIGHT      ( +60° to +120°)",
            5: "REAR-RIGHT (+120° to +180°)",
        }
        for s in range(self.n_sectors):
            if s in sector_data:
                dist, ang = sector_data[s]
                in_zone = "⚠ IN BARRIER ZONE" if dist < self.influence * self.d_safe else ""
                lines.append(
                    f"  Sector {s} {sector_names.get(s, '')}: {dist:.3f}m at {ang:+.1f}° {in_zone}"
                )
            else:
                lines.append(f"  Sector {s} {sector_names.get(s, '')}: -- no obstacle --")
        lines.append("")
        lines.append("── 6 OBSTACLES SENT TO NMPC SOLVER ──")
        lines.append(f"  d_safe={self.d_safe}m | barrier zone={self.influence * self.d_safe:.3f}m")
        lines.append("")
        if not selected:
            lines.append("  NO obstacles within range!")
        else:
            for i, (sector, (ox, oy, dist, ang_deg)) in enumerate(
                sorted(selected.items(), key=lambda x: x[1][2])
            ):
                direction = self.angle_to_label(ang_deg)
                in_zone = dist < self.influence * self.d_safe
                danger = "⚠ BARRIER ACTIVE" if in_zone else "✓ safe"
                lines.append(
                    f"  Obs {i + 1} (sector {sector}): dist={dist:.3f}m | angle={ang_deg:+.1f}° | {direction} | {danger}"
                )
                lines.append(f"         world=({ox:.2f},{oy:.2f})")
            for i in range(len(selected), self.n_sectors):
                lines.append(f"  Obs {i + 1}: -- dummy (999, 999) --")
        lines.append("")
        left_dist = sector_data.get(1, (999, 0))[0]
        right_dist = sector_data.get(4, (999, 0))[0]
        front_dist = min(sector_data.get(2, (999, 0))[0], sector_data.get(3, (999, 0))[0])
        lines.append("── CORRIDOR ANALYSIS ──")
        lines.append(
            f"  LEFT wall:  {left_dist:.3f}m {('⚠ too close' if left_dist < self.d_safe else '✓ ok')}"
        )
        lines.append(
            f"  RIGHT wall: {right_dist:.3f}m {('⚠ too close' if right_dist < self.d_safe else '✓ ok')}"
        )
        lines.append(
            f"  FRONT:      {front_dist:.3f}m {('⚠ blocked' if front_dist < self.d_safe else '✓ clear')}"
        )
        corridor_width = left_dist + right_dist
        lines.append(f"  Estimated corridor width seen by lidar: {corridor_width:.3f}m")
        lines.append(f"  Robot width (wheels): 0.67m")
        lines.append(f"  Available clearance per side: {(corridor_width - 0.67) / 2:.3f}m")
        lines.append("=" * 60)
        self.get_logger().info("\n" + "\n".join(lines))


def main():
    rclpy.init()
    node = ObstacleDebug()
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
