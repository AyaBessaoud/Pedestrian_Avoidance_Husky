#!/usr/bin/env python3
import statistics
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PointStamped, Point
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray
from people_msgs.msg import People, Person
from tf2_ros import Buffer, TransformListener
import tf2_geometry_msgs

INPUT_PEOPLE_TOPIC = "/people"
INPUT_SCAN_TOPIC = "/a200_0846/sensors/lidar2d_0/scan"
OUTPUT_PEOPLE_TOPIC = "/people_lidar_extended"
OUTPUT_VIZ_TOPIC = "/people_lidar_extended_viz"


def filter_outliers(values):
    if len(values) < 2:
        return values
    mu = statistics.mean(values)
    std = statistics.stdev(values)
    return [v for v in values if abs(v - mu) < std] or values


def get_best_value(values, technique="median"):
    if not values:
        return None
    if technique == "closest":
        return min(values, key=abs)
    elif technique == "average":
        return statistics.mean(values)
    else:
        return statistics.median(sorted(values))


class PersonLidarCarryover(Node):

    def __init__(self):
        super().__init__("person_lidar_carryover_node")
        self.declare_parameter("camera_timeout_sec", 0.6)
        self.declare_parameter("max_hold_sec", 2.0)
        self.declare_parameter("search_radius_m", 0.6)
        self.declare_parameter("cluster_gap_m", 0.2)
        self.declare_parameter("cluster_min_size_m", 0.1)
        self.declare_parameter("cluster_max_size_m", 0.9)
        self.camera_timeout_sec = self.get_parameter("camera_timeout_sec").value
        self.max_hold_sec = self.get_parameter("max_hold_sec").value
        self.search_radius_m = self.get_parameter("search_radius_m").value
        self.cluster_gap_m = self.get_parameter("cluster_gap_m").value
        self.cluster_min_size_m = self.get_parameter("cluster_min_size_m").value
        self.cluster_max_size_m = self.get_parameter("cluster_max_size_m").value
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        scan_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=5)
        self.create_subscription(People, INPUT_PEOPLE_TOPIC, self.people_cb, 10)
        self.create_subscription(LaserScan, INPUT_SCAN_TOPIC, self.scan_cb, scan_qos)
        self.people_pub = self.create_publisher(People, OUTPUT_PEOPLE_TOPIC, 10)
        self.viz_pub = self.create_publisher(MarkerArray, OUTPUT_VIZ_TOPIC, 10)
        self.anchor_pos_map = None
        self.anchor_time = None
        self.anchor_source = None
        self.last_people_msg_time = 0.0
        self.get_logger().info("=" * 60)
        self.get_logger().info("Person LiDAR carryover node ready.")
        self.get_logger().info(f"  camera in : {INPUT_PEOPLE_TOPIC}")
        self.get_logger().info(f"  scan in   : {INPUT_SCAN_TOPIC}")
        self.get_logger().info(f"  output    : {OUTPUT_PEOPLE_TOPIC}")
        self.get_logger().info("=" * 60)

    def people_cb(self, msg):
        if len(msg.people) == 0:
            return
        pos_map = self._closest_person_to_anchor_map(msg)
        if pos_map is None:
            return
        now = self._now_sec()
        self.anchor_pos_map = pos_map
        self.anchor_time = now
        self.anchor_source = "camera"
        self.last_people_msg_time = now
        self._publish_person(pos_map, source="camera", stamp=msg.header.stamp)

    def _closest_person_to_anchor_map(self, msg):
        best_pos = None
        best_dist = float("inf")
        for person in msg.people:
            try:
                pt = PointStamped()
                pt.header.frame_id = msg.header.frame_id
                pt.header.stamp = rclpy.time.Time().to_msg()
                pt.point.x = person.position.x
                pt.point.y = person.position.y
                pt.point.z = person.position.z
                pt_map = self.tf_buffer.transform(pt, "map")
                pos = (pt_map.point.x, pt_map.point.y)
            except Exception:
                continue
            if self.anchor_pos_map is None:
                return pos
            d = np.hypot(pos[0] - self.anchor_pos_map[0], pos[1] - self.anchor_pos_map[1])
            if d < best_dist:
                best_dist = d
                best_pos = pos
        return best_pos

    def scan_cb(self, msg):
        if self.anchor_pos_map is None:
            return
        now = self._now_sec()
        since_camera = now - self.last_people_msg_time
        since_anchor = now - self.anchor_time
        if since_camera < self.camera_timeout_sec:
            return
        if since_anchor > self.max_hold_sec:
            self.get_logger().warn(
                f"Anchor stale ({since_anchor:.1f}s > {self.max_hold_sec}s) — giving up.",
                throttle_duration_sec=1.0,
            )
            return
        self.get_logger().info(
            f"Camera quiet {since_camera:.1f}s — searching LiDAR near anchor ({self.anchor_pos_map[0]:.2f}, {self.anchor_pos_map[1]:.2f})...",
            throttle_duration_sec=0.5,
        )
        cluster_map = self._find_matching_cluster(msg)
        if cluster_map is None:
            return
        self.get_logger().info(
            f"MATCH: LiDAR cluster at ({cluster_map[0]:.2f}, {cluster_map[1]:.2f})",
            throttle_duration_sec=0.5,
        )
        self.anchor_pos_map = cluster_map
        self.anchor_time = now
        self.anchor_source = "lidar"
        self._publish_person(cluster_map, source="lidar", stamp=msg.header.stamp)

    def _find_matching_cluster(self, scan_msg):
        ranges = np.array(scan_msg.ranges)
        angles = scan_msg.angle_min + np.arange(len(ranges)) * scan_msg.angle_increment
        valid = (ranges > 0.05) & (ranges < 20.0) & np.isfinite(ranges)
        if not np.any(valid):
            self.get_logger().warn(
                "No valid scan points at all this frame.", throttle_duration_sec=1.0
            )
            return None
        r = ranges[valid]
        a = angles[valid]
        idx = np.where(valid)[0]
        lx = r * np.cos(a)
        ly = r * np.sin(a)
        clusters = []
        current = [(lx[0], ly[0])]
        for i in range(1, len(lx)):
            gap_in_index = idx[i] - idx[i - 1]
            d = np.hypot(lx[i] - lx[i - 1], ly[i] - ly[i - 1])
            if gap_in_index <= 3 and d <= self.cluster_gap_m:
                current.append((lx[i], ly[i]))
            else:
                clusters.append(current)
                current = [(lx[i], ly[i])]
        clusters.append(current)
        anchor_pt = PointStamped()
        anchor_pt.header.frame_id = "map"
        anchor_pt.header.stamp = rclpy.time.Time().to_msg()
        anchor_pt.point.x = self.anchor_pos_map[0]
        anchor_pt.point.y = self.anchor_pos_map[1]
        anchor_pt.point.z = 0.0
        try:
            anchor_in_scan = self.tf_buffer.transform(anchor_pt, scan_msg.header.frame_id)
        except Exception as e:
            self.get_logger().warn(
                f"TF map->{scan_msg.header.frame_id} failed: {e}", throttle_duration_sec=1.0
            )
            return None
        ax, ay = (anchor_in_scan.point.x, anchor_in_scan.point.y)
        best_centroid = None
        best_dist = float("inf")
        n_size_rejected = 0
        for pts in clusters:
            pts_arr = np.array(pts)
            width = np.hypot(
                pts_arr[:, 0].max() - pts_arr[:, 0].min(), pts_arr[:, 1].max() - pts_arr[:, 1].min()
            )
            if not self.cluster_min_size_m <= width <= self.cluster_max_size_m:
                n_size_rejected += 1
                continue
            cx = get_best_value(filter_outliers(list(pts_arr[:, 0])))
            cy = get_best_value(filter_outliers(list(pts_arr[:, 1])))
            if cx is None or cy is None:
                continue
            d = np.hypot(cx - ax, cy - ay)
            if d < best_dist:
                best_dist = d
                best_centroid = (cx, cy)
        if best_centroid is None:
            self.get_logger().warn(
                f"No cluster passed the size filter ({self.cluster_min_size_m}-{self.cluster_max_size_m}m); {n_size_rejected}/{len(clusters)} clusters rejected on size.",
                throttle_duration_sec=1.0,
            )
            return None
        if best_dist > self.search_radius_m:
            self.get_logger().warn(
                f"Closest cluster was {best_dist:.2f}m away — outside search_radius_m={self.search_radius_m}. Consider raising it.",
                throttle_duration_sec=1.0,
            )
            return None
        pt = PointStamped()
        pt.header.frame_id = scan_msg.header.frame_id
        pt.header.stamp = rclpy.time.Time().to_msg()
        pt.point.x, pt.point.y, pt.point.z = (best_centroid[0], best_centroid[1], 0.0)
        try:
            pt_map = self.tf_buffer.transform(pt, "map")
        except Exception:
            return None
        return (pt_map.point.x, pt_map.point.y)

    def _publish_person(self, pos_map, source, stamp):
        msg = People()
        msg.header.frame_id = "map"
        msg.header.stamp = stamp
        p = Person()
        p.name = "tracked_person"
        p.position = Point(x=pos_map[0], y=pos_map[1], z=0.0)
        p.velocity = Point(x=0.0, y=0.0, z=0.0)
        p.reliability = 1.0 if source == "camera" else 0.7
        p.tags = [source]
        msg.people = [p]
        self.people_pub.publish(msg)
        self._publish_viz(pos_map, source)

    def _publish_viz(self, pos_map, source):
        marker = Marker()
        marker.header.frame_id = "map"
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "person_lidar_carryover"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = pos_map[0]
        marker.pose.position.y = pos_map[1]
        marker.pose.position.z = 0.4
        marker.scale.x = marker.scale.y = marker.scale.z = 0.4
        marker.lifetime.sec = 1
        marker.lifetime.nanosec = 0
        if source == "camera":
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = (0.12, 0.47, 0.85, 0.9)
        else:
            marker.color.r, marker.color.g, marker.color.b, marker.color.a = (1.0, 0.5, 0.31, 0.9)
        arr = MarkerArray()
        arr.markers = [marker]
        self.viz_pub.publish(arr)

    def _now_sec(self):
        return self.get_clock().now().nanoseconds / 1000000000.0


def main():
    rclpy.init()
    node = PersonLidarCarryover()
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
