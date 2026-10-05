import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped, Point, PointStamped
from std_msgs.msg import Bool
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Path
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener
from people_msgs.msg import People
import tf2_geometry_msgs
import casadi as ca
import numpy as np


class HuskyNMPC(Node):

    def __init__(self):
        super().__init__("husky_nmpc_node")
        self.declare_parameter("w_pos", 49.8936)
        self.declare_parameter("w_heading", 2.4695)
        self.declare_parameter("w_vel", 0.7654)
        self.declare_parameter("w_smooth", 0.2909)
        self.declare_parameter("w_obs", 48.7617)
        self.declare_parameter("w_terminal_pos", 190.2921)
        self.declare_parameter("w_terminal_h", 4.8013)
        self.w_pos = self.get_parameter("w_pos").value
        self.w_heading = self.get_parameter("w_heading").value
        self.w_vel = self.get_parameter("w_vel").value
        self.w_smooth = self.get_parameter("w_smooth").value
        self.w_obs = self.get_parameter("w_obs").value
        self.w_terminal_pos = self.get_parameter("w_terminal_pos").value
        self.w_terminal_h = self.get_parameter("w_terminal_h").value
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        scan_qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, depth=5)
        self.cmd_pub = self.create_publisher(Twist, "/a200_0846/nmpc/cmd_vel", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/nmpc_viz", 10)
        self.active_state_pub = self.create_publisher(Bool, "/a200_0846/nmpc/active", 10)
        self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self.initial_pose_cb, 10
        )
        self.create_subscription(
            LaserScan, "/a200_0846/sensors/lidar2d_0/scan", self.scan_cb, scan_qos
        )
        self.create_subscription(Path, "/a200_0846/plan", self.path_cb, 10)
        self.create_subscription(People, "/people", self.pedestrian_cb, 10)
        self.is_localized = False
        self.state_now = [0.0, 0.0, 0.0]
        self.obstacles = []
        self.prev_solution = None
        self.loop_count = 0
        self.stuck_counter = 0
        self.last_position = [0.0, 0.0]
        self.solver_fail_count = 0
        self.global_path = []
        self.nmpc_active = False
        self.fixed_carrot = None
        self.last_crossing_point = None
        self.person_pos_map = None
        self.person_vel_map = None
        self.last_good_vel = np.array([0.0, 0.0, 0.0])
        self.prediction_time = 2.0
        self.person_last_seen = None
        self.person_hold_sec = 2.0
        self.person_obs_radius = 0.5
        self.prev_person_pos = None
        self.prev_person_time = None
        self.teleport_margin = 0.3
        self.teleport_reject_count = 0
        self.teleport_relock_after = 5
        self.coherent_frames = 0
        self.min_coherent_frames = 3
        self.coherence_gap_sec = 2.5
        self.activation_distance = 4.0
        self.path_threat_distance = 3.0
        self.min_person_speed = 0.1
        self.max_human_speed = 3.0
        self.max_vertical_vel = 0.8
        self.person_log_count = 0
        self.person_log_interval = 10
        self.carrot_lookahead = 5.0
        self.carrot_reach_dist = 0.4
        self.carrot_past_person = 2.0
        self.carrot_min_dist_from_robot = 1.5
        self.dt = 0.1
        self.N = 30
        self.num_obs = 10
        self.max_v = 1.0
        self.max_w = 0.85
        self.d_safe = 0.45
        self.influence_mult = 1.3
        self.min_obs_dist_this_activation = float("inf")
        self.setup_solver()
        self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("=" * 60)
        self.get_logger().info("NMPC | predictive | closest-person | velocity-filtered")
        self.get_logger().info(
            f"  activation_distance={self.activation_distance}m | path_threat_distance={self.path_threat_distance}m | min_person_speed={self.min_person_speed}m/s | max_human_speed={self.max_human_speed}m/s"
        )
        self.get_logger().info("  WEIGHTS (Optuna best, approx for bounded cost):")
        self.get_logger().info(
            f"    w_pos={self.w_pos} w_heading={self.w_heading} w_vel={self.w_vel} w_smooth={self.w_smooth}"
        )
        self.get_logger().info(
            f"    w_obs={self.w_obs} w_terminal_pos={self.w_terminal_pos} w_terminal_h={self.w_terminal_h}"
        )
        self.get_logger().info("=" * 60)

    def _filter_velocity(self, raw_vel):
        horiz_speed = float(np.linalg.norm(raw_vel[:2]))
        vert = abs(float(raw_vel[2]))
        if horiz_speed > self.max_human_speed or vert > self.max_vertical_vel:
            return (self.last_good_vel.copy(), False)
        good = np.array([raw_vel[0], raw_vel[1], 0.0])
        self.last_good_vel = good.copy()
        return (good, True)

    def _select_closest_person(self, msg):
        best_pos = None
        best_vel = None
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
                pos_map = np.array([pt_map.point.x, pt_map.point.y, pt_map.point.z])
                t = self.tf_buffer.lookup_transform("map", msg.header.frame_id, rclpy.time.Time())
                q = t.transform.rotation
                rot = np.array(
                    [
                        [
                            1 - 2 * (q.y**2 + q.z**2),
                            2 * (q.x * q.y - q.w * q.z),
                            2 * (q.x * q.z + q.w * q.y),
                        ],
                        [
                            2 * (q.x * q.y + q.w * q.z),
                            1 - 2 * (q.x**2 + q.z**2),
                            2 * (q.y * q.z - q.w * q.x),
                        ],
                        [
                            2 * (q.x * q.z - q.w * q.y),
                            2 * (q.y * q.z + q.w * q.x),
                            1 - 2 * (q.x**2 + q.y**2),
                        ],
                    ]
                )
                vel_map = rot @ np.array([person.velocity.x, person.velocity.y, person.velocity.z])
                d = np.hypot(pos_map[0] - self.state_now[0], pos_map[1] - self.state_now[1])
                if d < best_dist:
                    best_dist = d
                    best_pos = pos_map
                    best_vel = vel_map
            except Exception:
                continue
        return (best_pos, best_vel)

    def _is_person_threat(self, person_pos, person_vel, person_speed):
        if not self.global_path or person_pos is None:
            return False
        if person_speed < self.min_person_speed:
            return False
        x, y, _ = self.state_now
        px, py = (person_pos[0], person_pos[1])
        if np.hypot(px - x, py - y) > self.activation_distance:
            return False
        robot_path_idx = min(
            range(len(self.global_path)),
            key=lambda i: np.hypot(self.global_path[i][0] - x, self.global_path[i][1] - y),
        )
        vx, vy = (person_vel[0], person_vel[1])
        for t in (0.0, 0.5, 1.0, 1.5, 2.0):
            ppx = px + vx * t
            ppy = py + vy * t
            person_path_idx = min(
                range(len(self.global_path)),
                key=lambda i: np.hypot(self.global_path[i][0] - ppx, self.global_path[i][1] - ppy),
            )
            min_dist = np.hypot(
                self.global_path[person_path_idx][0] - ppx,
                self.global_path[person_path_idx][1] - ppy,
            )
            if person_path_idx >= robot_path_idx and min_dist <= self.path_threat_distance:
                self.last_crossing_point = (
                    self.global_path[person_path_idx][0],
                    self.global_path[person_path_idx][1],
                )
                return True
        return False

    def initial_pose_cb(self, msg):
        self.is_localized = True
        self.get_logger().info(
            f"[LOCALIZED] via RViz ({msg.pose.pose.position.x:.2f}, {msg.pose.pose.position.y:.2f})"
        )

    def path_cb(self, msg):
        self.global_path = [(p.pose.position.x, p.pose.position.y) for p in msg.poses]
        if self.loop_count % 20 == 0:
            self.get_logger().info(f"[PATH] {len(self.global_path)} waypoints from Nav2")

    def pedestrian_cb(self, msg):
        was_active = self.nmpc_active
        self.person_log_count += 1
        do_log = self.person_log_count % self.person_log_interval == 0
        person_speed = 0.0
        if len(msg.people) > 0:
            pos_map, raw_vel = self._select_closest_person(msg)
            if pos_map is None:
                if do_log:
                    self.get_logger().warn("[NMPC] TF failed for all people this frame")
                return
            now = self.get_clock().now()
            if self.prev_person_pos is not None and self.prev_person_time is not None:
                dt_s = (now - self.prev_person_time).nanoseconds * 1e-09
                gap_restart = dt_s > self.coherence_gap_sec
                jump = float(
                    np.hypot(
                        pos_map[0] - self.prev_person_pos[0], pos_map[1] - self.prev_person_pos[1]
                    )
                )
                max_jump = self.max_human_speed * max(dt_s, 0.001) + self.teleport_margin
                if not gap_restart and jump > max_jump:
                    self.teleport_reject_count += 1
                    self.coherent_frames = 0
                    if self.teleport_reject_count < self.teleport_relock_after:
                        if do_log:
                            self.get_logger().warn(
                                f"[TELEPORT] rejected jump={jump:.2f}m in {dt_s:.2f}s (max={max_jump:.2f}m) — holding last coherent pos ({self.teleport_reject_count}/{self.teleport_relock_after})"
                            )
                        return
                    else:
                        self.get_logger().warn(
                            f"[TELEPORT] {self.teleport_reject_count} consecutive impossible jumps — accepting new position (re-lock)"
                        )
                if gap_restart:
                    self.coherent_frames = 0
            self.teleport_reject_count = 0
            self.coherent_frames += 1
            self.prev_person_pos = (float(pos_map[0]), float(pos_map[1]))
            self.prev_person_time = now
            self.person_pos_map = pos_map
            self.person_last_seen = self.get_clock().now()
            filtered_vel, accepted = self._filter_velocity(raw_vel)
            self.person_vel_map = filtered_vel
            person_speed = float(np.linalg.norm(self.person_vel_map[:2]))
            if do_log:
                dist = np.hypot(
                    self.person_pos_map[0] - self.state_now[0],
                    self.person_pos_map[1] - self.state_now[1],
                )
                status = "MOVING" if person_speed >= self.min_person_speed else "STATIC"
                raw_speed = float(np.linalg.norm(raw_vel[:2]))
                rej = "" if accepted else f" [VEL REJECTED raw={raw_speed:.2f}]"
                self.get_logger().info(
                    f"[PERSON] {status} | pos=({self.person_pos_map[0]:.2f},{self.person_pos_map[1]:.2f}) | speed={person_speed:.2f}m/s{rej} | dist_to_robot={dist:.2f}m"
                )
            if not self.nmpc_active:
                if self.coherent_frames < self.min_coherent_frames:
                    if do_log:
                        self.get_logger().info(
                            f"[PERSIST] {self.coherent_frames}/{self.min_coherent_frames} coherent frames — not eligible to activate yet"
                        )
                elif self._is_person_threat(self.person_pos_map, self.person_vel_map, person_speed):
                    if len(self.global_path) > 0:
                        self.nmpc_active = True
                    else:
                        self.get_logger().warn("[NMPC] Person is threat but no path yet")
                elif do_log:
                    dist = np.hypot(
                        self.person_pos_map[0] - self.state_now[0],
                        self.person_pos_map[1] - self.state_now[1],
                    )
                    self.get_logger().info(
                        f"[NMPC] Not activating — speed={person_speed:.2f} | dist={dist:.2f}m"
                    )
        elif do_log:
            if self.nmpc_active:
                self.get_logger().info("[NMPC] Person lost from view — continuing to carrot")
            else:
                self.get_logger().info("[PERSON] No detection")
        if self.nmpc_active and (not was_active):
            carrot = self._compute_carrot()
            if carrot is None:
                self.get_logger().warn("[NMPC] No valid carrot (>=min dist) — not activating")
                self.nmpc_active = False
                return
            self.fixed_carrot = carrot
            self.prev_solution = None
            self.min_obs_dist_this_activation = float("inf")
            self.get_logger().info("=" * 60)
            self.get_logger().info("[NMPC] ACTIVATED — person predicted to cross path")
            self.get_logger().info(
                f"  Carrot: ({carrot[0]:.3f}, {carrot[1]:.3f}) heading={np.degrees(carrot[2]):.1f}°"
            )
            self.get_logger().info(f"  Robot:  ({self.state_now[0]:.3f}, {self.state_now[1]:.3f})")
            if self.person_pos_map is not None:
                dist = np.hypot(
                    self.person_pos_map[0] - self.state_now[0],
                    self.person_pos_map[1] - self.state_now[1],
                )
                self.get_logger().info(
                    f"  Person: ({self.person_pos_map[0]:.2f}, {self.person_pos_map[1]:.2f}) dist={dist:.2f}m speed={person_speed:.2f}m/s vel=({self.person_vel_map[0]:.2f},{self.person_vel_map[1]:.2f})"
                )
            self.get_logger().info(
                f"  Carrot dist: {np.hypot(carrot[0] - self.state_now[0], carrot[1] - self.state_now[1]):.2f}m"
            )
            self.get_logger().info("=" * 60)
            self.active_state_pub.publish(Bool(data=True))
        elif not self.nmpc_active and was_active:
            self.fixed_carrot = None
            self.last_crossing_point = None
            self.prev_solution = None
            self.get_logger().info("=" * 60)
            self.get_logger().info("[NMPC] DEACTIVATED — Nav2 resumes")
            self.get_logger().info("=" * 60)
            self.active_state_pub.publish(Bool(data=False))

    def scan_cb(self, msg):
        tmp_obs = []
        valid_count = 0
        too_close = 0
        x, y, th = self.state_now
        cos_th, sin_th = (np.cos(th), np.sin(th))
        stride = 2
        r_min, r_max = (0.15, 2.5)
        LIDAR_OFFSET_X = -0.228
        LIDAR_OFFSET_Y = 0.0
        closest_raw = float("inf")
        for i in range(0, len(msg.ranges), stride):
            r = msg.ranges[i]
            if r < closest_raw and msg.range_min < r < msg.range_max:
                closest_raw = r
            if r_min < r < r_max:
                angle = msg.angle_min + i * msg.angle_increment
                lx = r * np.cos(angle)
                ly = r * np.sin(angle)
                bx = lx + LIDAR_OFFSET_X
                by = ly + LIDAR_OFFSET_Y
                tmp_obs.append([x + cos_th * bx - sin_th * by, y + sin_th * bx + cos_th * by])
                valid_count += 1
                if r < 0.7:
                    too_close += 1
        self.obstacles = tmp_obs
        if self.loop_count % 10 == 0 and valid_count > 0:
            cr = f"{closest_raw:.2f}m" if closest_raw != float("inf") else "none"
            self.get_logger().info(
                f"[SCAN] {valid_count} pts | {too_close} VERY CLOSE (<0.7m) | closest_raw={cr}"
            )

    def _compute_carrot(self):
        if not self.global_path:
            return None
        x, y, _ = self.state_now
        robot_path_idx = min(
            range(len(self.global_path)),
            key=lambda i: np.hypot(self.global_path[i][0] - x, self.global_path[i][1] - y),
        )
        if self.person_pos_map is not None:
            if self.person_vel_map is not None and np.linalg.norm(self.person_vel_map[:2]) > 0.1:
                predicted_x = self.person_pos_map[0] + self.person_vel_map[0] * self.prediction_time
                predicted_y = self.person_pos_map[1] + self.person_vel_map[1] * self.prediction_time
            else:
                predicted_x, predicted_y = (self.person_pos_map[0], self.person_pos_map[1])
            self.get_logger().info(
                f"[CARROT] Person ACTUAL=({self.person_pos_map[0]:.2f},{self.person_pos_map[1]:.2f}) vel=({self.person_vel_map[0]:.2f},{self.person_vel_map[1]:.2f}) PREDICTED=({predicted_x:.2f},{predicted_y:.2f})"
            )
            person_path_idx = min(
                range(len(self.global_path)),
                key=lambda i: np.hypot(
                    self.global_path[i][0] - predicted_x, self.global_path[i][1] - predicted_y
                ),
            )
            self.get_logger().info(
                f"[CARROT] Robot idx={robot_path_idx}, Person crossing idx={person_path_idx}"
            )
            start = max(robot_path_idx + 1, person_path_idx)
            for i in range(start, len(self.global_path)):
                px, py = self.global_path[i]
                if (
                    np.hypot(px - x, py - y) >= self.carrot_min_dist_from_robot
                    and np.hypot(px - predicted_x, py - predicted_y) >= self.carrot_past_person
                ):
                    self.get_logger().info(
                        f"[CARROT] Selected: ({px:.2f}, {py:.2f}) dist_from_robot={np.hypot(px - x, py - y):.2f}m dist_past_person={np.hypot(px - predicted_x, py - predicted_y):.2f}m"
                    )
                    return [px, py, np.arctan2(py - y, px - x)]
            for i in range(len(self.global_path) - 1, robot_path_idx, -1):
                px, py = self.global_path[i]
                if np.hypot(px - x, py - y) >= self.carrot_min_dist_from_robot:
                    self.get_logger().warn(
                        f"[CARROT] No point past person — using farthest valid path point ({px:.2f},{py:.2f}) dist={np.hypot(px - x, py - y):.2f}m"
                    )
                    return [px, py, np.arctan2(py - y, px - x)]
            self.get_logger().warn(
                "[CARROT] No path point >= min dist from robot (near goal) — not activating"
            )
            return None
        for px, py in self.global_path[robot_path_idx:]:
            if np.hypot(px - x, py - y) >= self.carrot_lookahead:
                return [px, py, np.arctan2(py - y, px - x)]
        px, py = self.global_path[-1]
        if np.hypot(px - x, py - y) >= self.carrot_min_dist_from_robot:
            return [px, py, np.arctan2(py - y, px - x)]
        return None

    def setup_solver(self):
        x = ca.SX.sym("x")
        y = ca.SX.sym("y")
        th = ca.SX.sym("th")
        v = ca.SX.sym("v")
        w = ca.SX.sym("w")
        state = ca.vcat([x, y, th])
        ctrl = ca.vcat([v, w])
        f = ca.Function("f", [state, ctrl], [ca.vcat([v * ca.cos(th), v * ca.sin(th), w])])
        X = ca.SX.sym("X", 3, self.N + 1)
        U = ca.SX.sym("U", 2, self.N)
        self.P_person_pos = 6 + 2 * self.num_obs
        self.P_person_flag = self.P_person_pos + 2
        P = ca.SX.sym("P", self.P_person_flag + 2)
        cost = 0
        g = X[:, 0] - P[:3]

        def bounded_obs_cost(dx, dy, influence_r):
            d2 = dx * dx + dy * dy + 0.001
            d = ca.sqrt(d2)
            raw = self.w_obs * (self.d_safe * self.d_safe) / d2
            capped = ca.fmin(raw, self.w_obs)
            return ca.if_else(d < influence_r, capped, 0.0)

        for k in range(self.N):
            cost += self.w_pos * (X[0, k] - P[3]) ** 2 + self.w_pos * (X[1, k] - P[4]) ** 2
            eth = ca.atan2(ca.sin(X[2, k] - P[5]), ca.cos(X[2, k] - P[5]))
            cost += self.w_heading * eth**2 + self.w_vel * U[0, k] ** 2 + self.w_vel * U[1, k] ** 2
            if k > 0:
                cost += (
                    self.w_smooth * (U[0, k] - U[0, k - 1]) ** 2
                    + self.w_smooth * (U[1, k] - U[1, k - 1]) ** 2
                )
            g = ca.vcat([g, X[:, k + 1] - (X[:, k] + self.dt * f(X[:, k], U[:, k]))])
            for i in range(self.num_obs):
                ox, oy = (P[6 + 2 * i], P[6 + 2 * i + 1])
                cost += bounded_obs_cost(
                    X[0, k] - ox, X[1, k] - oy, self.d_safe * self.influence_mult
                )
            px_p = P[self.P_person_pos]
            py_p = P[self.P_person_pos + 1]
            flag = P[self.P_person_flag]
            prad = P[self.P_person_flag + 1]
            d2_p = (X[0, k] - px_p) ** 2 + (X[1, k] - py_p) ** 2 + 0.001
            raw_p = self.w_obs * (prad * prad) / d2_p
            capped_p = ca.fmin(raw_p, self.w_obs)
            cost += flag * ca.if_else(ca.sqrt(d2_p) < prad * self.influence_mult, capped_p, 0.0)
        cost += (
            self.w_terminal_pos * (X[0, self.N] - P[3]) ** 2
            + self.w_terminal_pos * (X[1, self.N] - P[4]) ** 2
        )
        cost += (
            self.w_terminal_h
            * ca.atan2(ca.sin(X[2, self.N] - P[5]), ca.cos(X[2, self.N] - P[5])) ** 2
        )
        self.solver = ca.nlpsol(
            "S",
            "ipopt",
            {"f": cost, "x": ca.vcat([ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)]), "g": g, "p": P},
            {
                "ipopt.print_level": 0,
                "print_time": 0,
                "ipopt.output_file": "/home/eya/nmpc_ipopt_debug.log",
                "ipopt.file_print_level": 5,
                "ipopt.max_iter": 350,
                "ipopt.warm_start_init_point": "yes",
                "ipopt.tol": 0.0001,
                "ipopt.acceptable_tol": 0.001,
                "ipopt.acceptable_iter": 5,
                "ipopt.mu_strategy": "adaptive",
            },
        )
        self.lbx = [-ca.inf] * (3 * (self.N + 1)) + [0.0, -self.max_w] * self.N
        self.ubx = [ca.inf] * (3 * (self.N + 1)) + [self.max_v, self.max_w] * self.N
        self.get_logger().info(
            f"[SOLVER] Ready | N={self.N} | num_obs={self.num_obs} | bounded cost + moving person"
        )

    def publish_viz(self, x_plan, sorted_obs=None):
        markers = MarkerArray()
        m = Marker()
        m.header.frame_id = "map"
        m.id = 999
        m.ns = "nmpc_trajectory"
        m.type = Marker.LINE_STRIP
        m.scale.x = 0.05
        m.color.g = 1.0
        m.color.a = 1.0
        for i in range(self.N + 1):
            p = Point()
            p.x = float(x_plan[0, i])
            p.y = float(x_plan[1, i])
            m.points.append(p)
        markers.markers.append(m)
        if self.fixed_carrot is not None:
            c = Marker()
            c.header.frame_id = "map"
            c.id = 1000
            c.ns = "carrot"
            c.type = Marker.SPHERE
            c.scale.x = c.scale.y = c.scale.z = 0.4
            c.color.r = 1.0
            c.color.a = 1.0
            c.pose.position.x = self.fixed_carrot[0]
            c.pose.position.y = self.fixed_carrot[1]
            c.pose.position.z = 0.5
            markers.markers.append(c)
        if self.person_pos_map is not None and self.person_vel_map is not None:
            speed = np.linalg.norm(self.person_vel_map[:2])
            pred_x = self.person_pos_map[0] + (
                self.person_vel_map[0] * self.prediction_time if speed > 0.1 else 0
            )
            pred_y = self.person_pos_map[1] + (
                self.person_vel_map[1] * self.prediction_time if speed > 0.1 else 0
            )
            pm = Marker()
            pm.header.frame_id = "map"
            pm.id = 1001
            pm.ns = "person_predicted_single"
            pm.type = Marker.SPHERE
            pm.scale.x = pm.scale.y = pm.scale.z = 0.3
            pm.color.b = 1.0
            pm.color.a = 0.8
            pm.pose.position.x = pred_x
            pm.pose.position.y = pred_y
            pm.pose.position.z = 0.5
            markers.markers.append(pm)
        if self.person_pos_map is not None:
            cur = Marker()
            cur.header.frame_id = "map"
            cur.id = 1002
            cur.ns = "person_current"
            cur.type = Marker.SPHERE
            cur.scale.x = cur.scale.y = cur.scale.z = 0.3
            cur.color.r = 1.0
            cur.color.g = 1.0
            cur.color.a = 0.9
            cur.pose.position.x = self.person_pos_map[0]
            cur.pose.position.y = self.person_pos_map[1]
            cur.pose.position.z = 0.5
            markers.markers.append(cur)
        if self.person_pos_map is not None and self.person_vel_map is not None:
            speed = np.linalg.norm(self.person_vel_map[:2])
            if speed > 0.1:
                arrow = Marker()
                arrow.header.frame_id = "map"
                arrow.id = 1003
                arrow.ns = "person_velocity"
                arrow.type = Marker.ARROW
                arrow.scale.x = 0.08
                arrow.scale.y = 0.15
                arrow.scale.z = 0.2
                arrow.color.r = 0.0
                arrow.color.g = 1.0
                arrow.color.b = 1.0
                arrow.color.a = 0.9
                start = Point()
                start.x = self.person_pos_map[0]
                start.y = self.person_pos_map[1]
                start.z = 0.5
                end = Point()
                arrow_len = min(speed, 2.0)
                end.x = self.person_pos_map[0] + self.person_vel_map[0] / speed * arrow_len
                end.y = self.person_pos_map[1] + self.person_vel_map[1] / speed * arrow_len
                end.z = 0.5
                arrow.points = [start, end]
                markers.markers.append(arrow)
        if self.person_pos_map is not None and self.person_vel_map is not None:
            speed = np.linalg.norm(self.person_vel_map[:2])
            if speed > 0.1:
                traj = Marker()
                traj.header.frame_id = "map"
                traj.id = 1004
                traj.ns = "person_predicted_path"
                traj.type = Marker.LINE_STRIP
                traj.scale.x = 0.04
                traj.color.r = 1.0
                traj.color.g = 0.5
                traj.color.a = 0.8
                for t in (0.0, 0.5, 1.0, 1.5, 2.0):
                    p = Point()
                    p.x = self.person_pos_map[0] + self.person_vel_map[0] * t
                    p.y = self.person_pos_map[1] + self.person_vel_map[1] * t
                    p.z = 0.45
                    traj.points.append(p)
                markers.markers.append(traj)
        if self.last_crossing_point is not None:
            cp = Marker()
            cp.header.frame_id = "map"
            cp.id = 1005
            cp.ns = "crossing_point"
            cp.type = Marker.CUBE
            cp.scale.x = cp.scale.y = cp.scale.z = 0.35
            cp.color.r = 1.0
            cp.color.b = 1.0
            cp.color.a = 0.9
            cp.pose.position.x = self.last_crossing_point[0]
            cp.pose.position.y = self.last_crossing_point[1]
            cp.pose.position.z = 0.5
            markers.markers.append(cp)
        if sorted_obs is not None:
            for i in range(min(self.num_obs, len(sorted_obs))):
                ob = Marker()
                ob.header.frame_id = "map"
                ob.id = 2000 + i
                ob.ns = "lidar_obstacles"
                ob.type = Marker.SPHERE
                ob.scale.x = ob.scale.y = ob.scale.z = 0.15
                ob.color.r = 0.0
                ob.color.g = 0.0
                ob.color.b = 0.0
                ob.color.a = 0.8
                ob.pose.position.x = sorted_obs[i][0]
                ob.pose.position.y = sorted_obs[i][1]
                ob.pose.position.z = 0.2
                markers.markers.append(ob)
        self.marker_pub.publish(markers)

    def check_if_stuck(self):
        movement = np.hypot(
            self.state_now[0] - self.last_position[0], self.state_now[1] - self.last_position[1]
        )
        self.last_position = [self.state_now[0], self.state_now[1]]
        self.stuck_counter = 0 if movement >= 0.01 else self.stuck_counter + 1
        return self.stuck_counter > 20

    def control_loop(self):
        self.loop_count += 1
        try:
            t = self.tf_buffer.lookup_transform("map", "base_link", rclpy.time.Time())
            self.state_now = [
                t.transform.translation.x,
                t.transform.translation.y,
                np.arctan2(
                    2
                    * (
                        t.transform.rotation.w * t.transform.rotation.z
                        + t.transform.rotation.x * t.transform.rotation.y
                    ),
                    1 - 2 * (t.transform.rotation.y**2 + t.transform.rotation.z**2),
                ),
            ]
            if not self.is_localized:
                self.is_localized = True
                self.get_logger().info(
                    f"[LOCALIZED] via TF ({self.state_now[0]:.2f}, {self.state_now[1]:.2f})"
                )
        except Exception:
            if self.loop_count % 50 == 0:
                self.get_logger().warn("[WAITING] No map→base_link TF yet")
            return
        if not self.nmpc_active:
            if self.loop_count % 50 == 0:
                self.get_logger().info(
                    "[IDLE] Nav2 driving | NMPC activates when person predicted to cross path"
                )
            return
        if self.fixed_carrot is None:
            self.get_logger().warn("[NMPC] Active but no carrot — deactivating")
            self.nmpc_active = False
            return
        cx, cy, c_heading = self.fixed_carrot
        dist_to_carrot = np.hypot(cx - self.state_now[0], cy - self.state_now[1])
        if dist_to_carrot < self.carrot_reach_dist:
            person_close = False
            if self.person_pos_map is not None and self.person_last_seen is not None:
                age = (self.get_clock().now() - self.person_last_seen).nanoseconds * 1e-09
                if age <= self.person_hold_sec:
                    pd = np.hypot(
                        self.person_pos_map[0] - self.state_now[0],
                        self.person_pos_map[1] - self.state_now[1],
                    )
                    person_close = pd < 1.5
            if person_close:
                if self.loop_count % 10 == 0:
                    self.get_logger().info(
                        "[NMPC] Carrot reached but person still within 1.5m — HOLDING at give-way point until they clear"
                    )
                self.cmd_pub.publish(Twist())
                return
            self.nmpc_active = False
            self.fixed_carrot = None
            self.prev_solution = None
            self.get_logger().info("=" * 60)
            self.get_logger().info(
                f"[NMPC] Carrot reached (dist={dist_to_carrot:.2f}m) — DEACTIVATING"
            )
            self.get_logger().info(
                f"[NMPC] Closest approach to obstacle during activation: {self.min_obs_dist_this_activation:.2f}m"
            )
            self.get_logger().info("=" * 60)
            self.active_state_pub.publish(Bool(data=False))
            return
        if self.global_path:
            gx, gy = self.global_path[-1]
            if np.hypot(gx - self.state_now[0], gy - self.state_now[1]) < self.carrot_reach_dist:
                self.nmpc_active = False
                self.fixed_carrot = None
                self.prev_solution = None
                self.get_logger().info("=" * 60)
                self.get_logger().info("[NMPC] Reached path end (goal) — DEACTIVATING")
                self.get_logger().info(
                    f"[NMPC] Closest approach to obstacle during activation: {self.min_obs_dist_this_activation:.2f}m"
                )
                self.get_logger().info("=" * 60)
                self.active_state_pub.publish(Bool(data=False))
                self.cmd_pub.publish(Twist())
                return
        sorted_obs = sorted(
            self.obstacles,
            key=lambda o: (o[0] - self.state_now[0]) ** 2 + (o[1] - self.state_now[1]) ** 2,
        )
        obs_to_pass = []
        closest_obs_dist = float("inf")
        for i in range(self.num_obs):
            if i < len(sorted_obs):
                obs_to_pass.extend(sorted_obs[i])
                d = np.hypot(
                    sorted_obs[i][0] - self.state_now[0], sorted_obs[i][1] - self.state_now[1]
                )
                closest_obs_dist = min(closest_obs_dist, d)
            else:
                obs_to_pass.extend([100.0, 100.0])
        if closest_obs_dist < self.min_obs_dist_this_activation:
            self.min_obs_dist_this_activation = closest_obs_dist
        person_px, person_py = (100.0, 100.0)
        person_flag = 0.0
        if self.person_pos_map is not None and self.person_last_seen is not None:
            age = (self.get_clock().now() - self.person_last_seen).nanoseconds * 1e-09
            if age <= self.person_hold_sec:
                person_px = float(self.person_pos_map[0])
                person_py = float(self.person_pos_map[1])
                person_flag = 1.0
        if self.loop_count % 5 == 0:
            in_zone = closest_obs_dist < self.d_safe * self.influence_mult
            self.get_logger().info("-" * 50)
            self.get_logger().info(f"[LOOP {self.loop_count}] NMPC ACTIVE")
            self.get_logger().info(
                f"  Robot:  ({self.state_now[0]:.3f}, {self.state_now[1]:.3f}, {np.degrees(self.state_now[2]):.1f}°)"
            )
            self.get_logger().info(f"  Carrot: ({cx:.3f}, {cy:.3f}) dist={dist_to_carrot:.2f}m")
            p_state = "ENFORCED" if person_flag > 0.5 else "stale/none"
            self.get_logger().info(f"  Person-obs: ({person_px:.2f}, {person_py:.2f}) [{p_state}]")
            self.get_logger().info(
                f"  closest_obs={closest_obs_dist:.2f}m {('IN ZONE' if in_zone else 'safe')} | min_so_far={self.min_obs_dist_this_activation:.2f}m"
            )
        if self.check_if_stuck() and self.loop_count % 10 == 0:
            self.get_logger().error(f"[STUCK!] No movement for {self.stuck_counter} cycles")
            self.prev_solution = None
        p_val = (
            self.state_now
            + [cx, cy, c_heading]
            + obs_to_pass
            + [person_px, person_py, person_flag, self.person_obs_radius]
        )
        if self.prev_solution is not None:
            x0 = self.prev_solution
        else:
            v_seed = min(0.4, getattr(self, "max_v", 0.8))
            dx_c = cx - self.state_now[0]
            dy_c = cy - self.state_now[1]
            th_c = np.arctan2(dy_c, dx_c)
            err = np.arctan2(np.sin(th_c - self.state_now[2]), np.cos(th_c - self.state_now[2]))
            w_seed = max(-self.max_w, min(self.max_w, 1.5 * err))
            x0 = np.zeros(3 * (self.N + 1) + 2 * self.N)
            xs, ys, ths = (self.state_now[0], self.state_now[1], self.state_now[2])
            for k in range(self.N + 1):
                x0[3 * k], x0[3 * k + 1], x0[3 * k + 2] = (xs, ys, ths)
                xs += v_seed * np.cos(ths) * self.dt
                ys += v_seed * np.sin(ths) * self.dt
                ths += w_seed * self.dt
            u_off = 3 * (self.N + 1)
            for k in range(self.N):
                x0[u_off + 2 * k] = v_seed
                x0[u_off + 2 * k + 1] = w_seed
        try:
            sol = self.solver(
                x0=x0,
                p=p_val,
                lbg=[0.0] * (3 * (self.N + 1)),
                ubg=[0.0] * (3 * (self.N + 1)),
                lbx=self.lbx,
                ubx=self.ubx,
            )
            stats = self.solver.stats()
            success = stats.get("success", False)
            return_status = stats.get("return_status", "UNKNOWN")
            if not success:
                self.solver_fail_count += 1
                self.get_logger().error(
                    f"[SOLVER] Failed (count={self.solver_fail_count}) status={return_status}"
                )
                STOP_RADIUS = 0.6
                if closest_obs_dist < STOP_RADIUS or self.solver_fail_count >= 4:
                    self.get_logger().warn(
                        f"[SOLVER] STOP (closest_obs={closest_obs_dist:.2f}m < {STOP_RADIUS}m or fails={self.solver_fail_count}) — holding position"
                    )
                    self.prev_solution = None
                    self.cmd_pub.publish(Twist())
                    return
                if self.prev_solution is not None:
                    u_prev = self.prev_solution[3 * (self.N + 1) :]
                    w_coast = 0.5 * float(u_prev[1])
                    if closest_obs_dist > 1.2:
                        v_coast = min(0.5 * abs(float(u_prev[0])) + 0.15, 0.4)
                    else:
                        v_coast = 0.0
                    self.get_logger().warn(
                        f"[SOLVER] Coasting (closest_obs={closest_obs_dist:.2f}m) v={v_coast:.2f} w={w_coast:.2f}"
                    )
                    cmd = Twist()
                    cmd.linear.x = v_coast
                    cmd.angular.z = w_coast
                    self.cmd_pub.publish(cmd)
                else:
                    self.cmd_pub.publish(Twist())
                return
            self.solver_fail_count = 0
            self.prev_solution = np.array(sol["x"]).flatten()
            x_sol = ca.reshape(sol["x"][: 3 * (self.N + 1)], 3, self.N + 1)
            u_sol = ca.reshape(sol["x"][3 * (self.N + 1) :], 2, self.N)
            v_cmd = float(u_sol[0, 0])
            w_cmd = float(u_sol[1, 0])
            v_cap = self.max_v
            if closest_obs_dist < 0.7:
                v_cap = 0.1
            elif closest_obs_dist < 1.0:
                v_cap = 0.25
            elif closest_obs_dist < 1.5:
                v_cap = 0.45
            if v_cmd > v_cap:
                if self.loop_count % 5 == 0:
                    self.get_logger().info(
                        f"  [GOVERNOR] v {v_cmd:.2f}->{v_cap:.2f} (closest_obs={closest_obs_dist:.2f}m)"
                    )
                v_cmd = v_cap
            if self.loop_count % 5 == 0:
                self.get_logger().info(
                    f"  [SOLVER] success={success} iters={stats.get('iter_count', -1)}"
                )
                self.get_logger().info(f"  [CMD] v={v_cmd:.4f} w={w_cmd:.4f}")
            self.publish_viz(x_sol, sorted_obs)
            cmd = Twist()
            cmd.linear.x = v_cmd
            cmd.angular.z = w_cmd
            self.cmd_pub.publish(cmd)
        except Exception as e:
            self.get_logger().error(f"[SOLVER EXCEPTION] {e}")
            self.prev_solution = None
            self.cmd_pub.publish(Twist())


def main():
    rclpy.init()
    node = HuskyNMPC()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
