#!/usr/bin/env python3
import json
import os
import time
from datetime import datetime
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from people_msgs.msg import People
from action_msgs.msg import GoalStatusArray, GoalStatus
from tf2_ros import Buffer, TransformListener

GOAL_STATUS_TOPIC = "/a200_0846/navigate_to_pose/_action/status"
POSE_TOPIC = "/a200_0846/amcl_pose"
PLAN_TOPIC = "/a200_0846/plan"
PEOPLE_TOPIC = "/people"
PEOPLE_EXTENDED_TOPIC = "/people_lidar_extended"
RAW_SCAN_TOPIC = "/a200_0846/sensors/lidar2d_0/scan"
EXECUTED_CMD_TOPIC = "/a200_0846/platform/cmd_vel_unstamped"
NMPC_CMD_TOPIC = "/a200_0846/nmpc/cmd_vel"
COLLISION_DIST = 0.3
RAW_COLLISION_DIST = 0.1
JERK_SAMPLE_INTERVAL = 0.1
RESULTS_FILE = os.path.expanduser("~/simple_metrics_results.json")
STATUS_NAMES = {
    GoalStatus.STATUS_UNKNOWN: "UNKNOWN",
    GoalStatus.STATUS_ACCEPTED: "ACCEPTED",
    GoalStatus.STATUS_EXECUTING: "EXECUTING",
    GoalStatus.STATUS_CANCELING: "CANCELING",
    GoalStatus.STATUS_SUCCEEDED: "SUCCEEDED",
    GoalStatus.STATUS_CANCELED: "CANCELED",
    GoalStatus.STATUS_ABORTED: "ABORTED",
}


class SimpleMetricsNode(Node):

    def __init__(self):
        super().__init__("simple_metrics_node")
        self.trial_num = 0
        self._seen_goal_ids = set()
        self._current_goal_id = None
        self.in_trial = False
        self._reset_trial_state()
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(GoalStatusArray, GOAL_STATUS_TOPIC, self.status_cb, 10)
        self.create_subscription(PoseWithCovarianceStamped, POSE_TOPIC, self.pose_cb, 10)
        self.create_subscription(Path, PLAN_TOPIC, self.plan_cb, 10)
        self.create_subscription(People, PEOPLE_TOPIC, self.people_camera_cb, 10)
        self.create_subscription(People, PEOPLE_EXTENDED_TOPIC, self.people_extended_cb, 10)
        self.create_subscription(Twist, EXECUTED_CMD_TOPIC, self.cmd_cb, 10)
        self.create_subscription(Twist, NMPC_CMD_TOPIC, self.nmpc_cmd_cb, 10)
        scan_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=5)
        self.create_subscription(LaserScan, RAW_SCAN_TOPIC, self.raw_scan_cb, scan_qos)
        self.get_logger().info("=" * 60)
        self.get_logger().info("Auto metrics logger ready.")
        self.get_logger().info(f"Listening for Nav2 goal status on: {GOAL_STATUS_TOPIC}")
        self.get_logger().info("Send a goal whenever you're ready.")
        self.get_logger().info(f"Results file: {RESULTS_FILE}")
        self.get_logger().info("=" * 60)

    def _reset_trial_state(self):
        self.reference_path = None
        self.robot_pos = None
        self.start_time = None
        self.last_pos_for_path = None
        self.path_length = 0.0
        self.max_path_deviation = 0.0
        self.max_ang_vel = 0.0
        self._prev_cmd_time = None
        self._prev_v = None
        self._prev_w = None
        self._prev_accel_v = None
        self._prev_accel_w = None
        self._jerk_v_sum = 0.0
        self._jerk_w_sum = 0.0
        self._jerk_count = 0
        self.max_linear_jerk = 0.0
        self.max_angular_jerk = 0.0
        self.person_pos = {"camera": None, "extended": None}
        self.min_person_dist = {"camera": float("inf"), "extended": float("inf")}
        self.min_obstacle_dist = float("inf")
        self.collision_count = 0
        self._currently_colliding = False
        self.nmpc_active = False
        self.give_way_triggered = False
        self.give_way_collision_happened = False
        self._last_nmpc_cmd_time = 0.0

    def status_cb(self, msg):
        for status in msg.status_list:
            gid = bytes(status.goal_info.goal_id.uuid)
            st = status.status
            if not self.in_trial:
                if gid not in self._seen_goal_ids and st in (
                    GoalStatus.STATUS_ACCEPTED,
                    GoalStatus.STATUS_EXECUTING,
                ):
                    self._start_trial(gid)
            elif gid == self._current_goal_id:
                if st in (
                    GoalStatus.STATUS_SUCCEEDED,
                    GoalStatus.STATUS_CANCELED,
                    GoalStatus.STATUS_ABORTED,
                ):
                    self._finish_trial(st)

    def _start_trial(self, gid):
        self.trial_num += 1
        self._current_goal_id = gid
        self._seen_goal_ids.add(gid)
        self.in_trial = True
        self._reset_trial_state()
        self.start_time = time.time()
        self.get_logger().info(f"\n>>> TRIAL {self.trial_num} started — recording...")

    def _finish_trial(self, final_status):
        self.in_trial = False
        self._seen_goal_ids.add(self._current_goal_id)
        summary = self.compute_summary(final_status)
        self._current_goal_id = None
        print("\n" + "=" * 60)
        print(f"TRIAL {summary['trial']} SUMMARY  (Nav2 result: {summary['goal_result']})")
        print("=" * 60)
        for k, v in summary.items():
            if k in ("trial", "goal_result", "time"):
                continue
            print(f"  {k:28s}: {v}")
        print("=" * 60)
        results = []
        if os.path.exists(RESULTS_FILE):
            try:
                with open(RESULTS_FILE) as f:
                    results = json.load(f)
                if not isinstance(results, list):
                    results = []
            except Exception:
                results = []
        results.append(summary)
        with open(RESULTS_FILE, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved. ({len(results)} trial(s) total in {RESULTS_FILE})\n")
        self.get_logger().info("Waiting for the next goal...")

    def plan_cb(self, msg):
        if not self.in_trial:
            return
        if self.reference_path is None and len(msg.poses) > 0:
            self.reference_path = np.array(
                [[p.pose.position.x, p.pose.position.y] for p in msg.poses]
            )

    def pose_cb(self, msg):
        if not self.in_trial:
            return
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.robot_pos = (x, y)
        if self.last_pos_for_path is not None:
            self.path_length += float(
                np.hypot(x - self.last_pos_for_path[0], y - self.last_pos_for_path[1])
            )
        self.last_pos_for_path = (x, y)
        if self.reference_path is not None:
            d = np.min(np.hypot(self.reference_path[:, 0] - x, self.reference_path[:, 1] - y))
            if d > self.max_path_deviation:
                self.max_path_deviation = float(d)
        self._check_person_distance("camera")
        self._check_person_distance("extended")

    def people_camera_cb(self, msg):
        if not self.in_trial or len(msg.people) == 0:
            return
        p = msg.people[0]
        self.person_pos["camera"] = (p.position.x, p.position.y)
        self._check_person_distance("camera")

    def people_extended_cb(self, msg):
        if not self.in_trial or len(msg.people) == 0:
            return
        p = msg.people[0]
        self.person_pos["extended"] = (p.position.x, p.position.y)
        self._check_person_distance("extended")

    def _check_person_distance(self, source):
        person_pos = self.person_pos[source]
        if self.robot_pos is None or person_pos is None:
            return
        d = float(np.hypot(self.robot_pos[0] - person_pos[0], self.robot_pos[1] - person_pos[1]))
        if d < self.min_person_dist[source]:
            self.min_person_dist[source] = d

    def cmd_cb(self, msg):
        if not self.in_trial:
            return
        v = float(msg.linear.x)
        w = float(msg.angular.z)
        if abs(w) > self.max_ang_vel:
            self.max_ang_vel = abs(w)
        now = self._now_sec()
        if self._prev_cmd_time is None:
            self._prev_cmd_time = now
            self._prev_v = v
            self._prev_w = w
            return
        dt = now - self._prev_cmd_time
        if dt < JERK_SAMPLE_INTERVAL:
            return
        accel_v = (v - self._prev_v) / dt
        accel_w = (w - self._prev_w) / dt
        if self._prev_accel_v is not None:
            jerk_v = (accel_v - self._prev_accel_v) / dt
            jerk_w = (accel_w - self._prev_accel_w) / dt
            self._jerk_v_sum += abs(jerk_v)
            self._jerk_w_sum += abs(jerk_w)
            self._jerk_count += 1
            if abs(jerk_v) > self.max_linear_jerk:
                self.max_linear_jerk = abs(jerk_v)
            if abs(jerk_w) > self.max_angular_jerk:
                self.max_angular_jerk = abs(jerk_w)
        self._prev_accel_v = accel_v
        self._prev_accel_w = accel_w
        self._prev_cmd_time = now
        self._prev_v = v
        self._prev_w = w

    def _now_sec(self):
        return self.get_clock().now().nanoseconds / 1000000000.0

    def raw_scan_cb(self, msg):
        if not self.in_trial:
            return
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        valid = (ranges > 0.05) & (ranges < 20.0) & np.isfinite(ranges)
        if not np.any(valid):
            return
        r = ranges[valid]
        a = angles[valid]
        lx = r * np.cos(a)
        ly = r * np.sin(a)
        try:
            t = self.tf_buffer.lookup_transform("base_link", msg.header.frame_id, rclpy.time.Time())
        except Exception as e:
            self.get_logger().warn(
                f"TF base_link->{msg.header.frame_id} failed: {e}", throttle_duration_sec=2.0
            )
            return
        tx, ty = (t.transform.translation.x, t.transform.translation.y)
        q = t.transform.rotation
        yaw = 2.0 * np.arctan2(q.z, q.w)
        cos_y, sin_y = (np.cos(yaw), np.sin(yaw))
        bx = lx * cos_y - ly * sin_y + tx
        by = lx * sin_y + ly * cos_y + ty
        d = float(np.min(np.hypot(bx, by)))
        if d < self.min_obstacle_dist:
            self.min_obstacle_dist = d
        is_colliding_now = d < RAW_COLLISION_DIST
        if is_colliding_now and (not self._currently_colliding):
            self.collision_count += 1
            if self.nmpc_active:
                self.give_way_collision_happened = True
        self._currently_colliding = is_colliding_now

    def nmpc_cmd_cb(self, msg):
        if not self.in_trial:
            return
        if abs(msg.linear.x) > 0.01 or abs(msg.angular.z) > 0.01:
            self.nmpc_active = True
            self.give_way_triggered = True
            self._last_nmpc_cmd_time = time.time()
        elif time.time() - self._last_nmpc_cmd_time > 0.5:
            self.nmpc_active = False

    def compute_summary(self, final_status):
        elapsed = time.time() - self.start_time if self.start_time else 0.0
        avg_velocity = self.path_length / elapsed if elapsed > 0 else 0.0
        if not self.give_way_triggered:
            give_way_success = "N/A (NMPC never activated — DWB-only or no threat)"
        else:
            give_way_success = not self.give_way_collision_happened

        def dist_or_none(source):
            d = self.min_person_dist[source]
            return round(d, 3) if d < float("inf") else None

        avg_linear_jerk = self._jerk_v_sum / self._jerk_count if self._jerk_count > 0 else None
        avg_angular_jerk = self._jerk_w_sum / self._jerk_count if self._jerk_count > 0 else None
        return {
            "trial": self.trial_num,
            "time": datetime.now().isoformat(),
            "goal_result": STATUS_NAMES.get(final_status, "UNKNOWN"),
            "time_to_goal_s": round(elapsed, 1),
            "max_path_deviation_m": round(self.max_path_deviation, 3),
            "average_velocity_mps": round(avg_velocity, 3),
            "max_angular_velocity_radps": round(self.max_ang_vel, 3),
            "avg_linear_jerk_mps3": (
                round(avg_linear_jerk, 3) if avg_linear_jerk is not None else None
            ),
            "max_linear_jerk_mps3": round(self.max_linear_jerk, 3),
            "avg_angular_jerk_radps3": (
                round(avg_angular_jerk, 3) if avg_angular_jerk is not None else None
            ),
            "max_angular_jerk_radps3": round(self.max_angular_jerk, 3),
            "give_way_success": give_way_success,
            "min_obstacle_distance_m": (
                round(self.min_obstacle_dist, 3) if self.min_obstacle_dist < float("inf") else None
            ),
            "number_of_collisions": self.collision_count,
            "extended_min_pedestrian_distance_m": dist_or_none("extended"),
            "camera_only_min_pedestrian_distance_m": dist_or_none("camera"),
        }


def main():
    rclpy.init()
    node = SimpleMetricsNode()
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
