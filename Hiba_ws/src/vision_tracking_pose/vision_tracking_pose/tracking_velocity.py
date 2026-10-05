#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from std_msgs.msg import Header
from geometry_msgs.msg import Point
from vision_tracking_pose_msgs.msg import DetectionArray, TrackedPerson, TrackedPeople
from people_msgs.msg import People, Person
import numpy as np
import time
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.utils import yaml_load, IterableSimpleNamespace
from pathlib import Path


class VelocityTrackingNode(Node):
    def __init__(self):
        super().__init__('tracking_node')
        self.get_logger().info("VelocityTrackingNode started.")

        config_path = str(Path.home() / "Downloads/Husky/Hiba_ws/src/vision_tracking_pose/vision_tracking_pose/botsort.yaml")
        cfg = yaml_load(config_path)
        args = IterableSimpleNamespace(**cfg)
        self.tracker = BYTETracker(args=args, frame_rate=13)
        self.get_logger().info(f"Loaded BYTETracker config from: {config_path}")

        self.sub_detections = self.create_subscription(
            DetectionArray,
            '/detections_3d',
            self.detections_callback,
            qos_profile_sensor_data
        )
        self.get_logger().info("Subscribed to /detections_3d")

        self.pub_tracked_person = self.create_publisher(TrackedPerson, '/TrackedPerson', 10)
        self.pub_tracked_people = self.create_publisher(TrackedPeople, '/TrackedPeople', 10)
        self.pub_people = self.create_publisher(People, '/people', 10)

        # ── FIX 2+3: world-frame positions & velocities ──────────────
        # Positions arrive in base_link (robot-relative). Differentiating
        # robot-relative positions contaminated velocities with the ROBOT's
        # own motion (robot turning 0.7rad/s made a person 4m away appear to
        # move 2.8 m/s -> the phantom velocity spikes). We now transform each
        # position to the world-fixed map frame BEFORE computing velocity.
        # SCARCITY-SAFE: TF lookup uses latest-available and, on failure,
        # falls back to the last cached transform (a one-frame-stale robot
        # pose shifts the position a few cm at most) — a detection is NEVER
        # dropped because of TF.
        self.declare_parameter("world_frame", "map")
        self.declare_parameter("source_frame", "base_link")
        self.world_frame = self.get_parameter("world_frame").value
        self.source_frame = self.get_parameter("source_frame").value
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.cached_tf = None   # last good (map <- base_link) transform

        # Track state per person ID:
        # track_history[id] = (prev_pos, prev_time, smoothed_velocity)
        self.track_history = {}

        # Max position jump allowed between frames (meters)
        # Real humans can't teleport — rejects bad depth readings
        self.max_position_jump = 1.0

    @staticmethod
    def _apply_tf(tf, p):
        """Rotate+translate point p (np.array[3]) by TransformStamped tf."""
        q = tf.transform.rotation
        t = tf.transform.translation
        qv = np.array([q.x, q.y, q.z])
        uv = np.cross(qv, p)
        uuv = np.cross(qv, uv)
        p_rot = p + 2 * (q.w * uv + uuv)
        return p_rot + np.array([t.x, t.y, t.z])

    def detections_callback(self, msg: DetectionArray):
        # FIX 4: use the MESSAGE STAMP, not wall clock. time.time() measured
        # wall seconds while positions move in SIM seconds — under Gazebo with
        # real-time-factor != 1 every velocity was scaled by the RTF. The
        # header stamp is sim time, matching the world the positions live in.
        # Fallback to wall clock only if the stamp is unset (zeros).
        stamp = msg.header.stamp
        if stamp.sec == 0 and stamp.nanosec == 0:
            current_time = time.time()
        else:
            current_time = stamp.sec + stamp.nanosec * 1e-9

        # FIX 2+3: refresh (or fall back to cached) map<-base_link transform
        try:
            self.cached_tf = self.tf_buffer.lookup_transform(
                self.world_frame, self.source_frame, rclpy.time.Time())
        except Exception:
            pass  # keep self.cached_tf as-is (last good); never drop detections
        self.get_logger().info(f" Received {len(msg.detections)} detections at time {current_time:.2f}")

        tracked_people_msg = TrackedPeople()
        tracked_people_msg.header = msg.header

        people_msg = People()
        people_msg.header = msg.header
        # FIX 3: the header previously LIED — it kept the camera optical frame
        # id while the data was base_link coordinates, so consumers (the NMPC)
        # applied the wrong rotation to every position. The header now states
        # the frame the data is actually in (set after the loop, see below).

        for det in msg.detections:
            self.get_logger().info(f" Processing detection ID: {det.id}")

            left_hip = None
            right_hip = None

            if not det.keypoints3d.data:
                self.get_logger().warn(f" Detection {det.id} has no keypoints3d data.")
                continue

            for kp in det.keypoints3d.data:
                self.get_logger().debug(
                    f" Keypoint ID={kp.id}, "
                    f"Point=({kp.point.x:.2f}, {kp.point.y:.2f}, {kp.point.z:.2f})")
                if kp.id == 12:
                    left_hip = kp.point
                elif kp.id == 13:
                    right_hip = kp.point

            if left_hip and right_hip:
                x = (left_hip.x + right_hip.x) / 2
                y = (left_hip.y + right_hip.y) / 2
                z = (left_hip.z + right_hip.z) / 2
            elif left_hip:
                x, y, z = left_hip.x, left_hip.y, left_hip.z
            elif right_hip:
                x, y, z = right_hip.x, right_hip.y, right_hip.z
            else:
                self.get_logger().warn(
                    f"No valid hips found in detection {det.id}, skipping.")
                continue

            current_pos = np.array([x, y, z])
            # FIX 2+3: express position in the world frame before velocity math
            out_frame = self.source_frame  # truthful fallback if TF never seen
            if self.cached_tf is not None:
                current_pos = self._apply_tf(self.cached_tf, current_pos)
                out_frame = self.world_frame
            x, y, z = float(current_pos[0]), float(current_pos[1]), float(current_pos[2])

            # Get track_id before any history lookup
            track_id = det.id or f"person_{det.class_id}"

            # --- Position jump rejection ---
            # Reject bad depth readings — real humans can't teleport 2m in one frame
            # This removes spikes caused by incorrect depth pixel values
            if track_id in self.track_history:
                prev_pos = self.track_history[track_id][0]
                jump = np.linalg.norm(current_pos - prev_pos)
                if jump > self.max_position_jump:
                    self.get_logger().warn(
                        f"[{track_id}] Bad depth — position jumped {jump:.2f}m, skipping")
                    continue

            velocity = Point(x=0.0, y=0.0, z=0.0)

            if track_id in self.track_history:
                prev_pos, prev_time, prev_smooth = self.track_history[track_id]
                dt = current_time - prev_time

                # MIN-DT GUARD (repairs a defect FIX 4 introduced): stamp-based
                # dt can be tiny or NEGATIVE (out-of-order stamps, observed
                # -135ms; 46/108 gaps <50ms under load). Dividing ~0.2m depth
                # jitter by a 30ms dt fabricates 6+ m/s. Only differentiate
                # once >=50ms of sim time has accumulated; below that, hold
                # the previous smoothed velocity AND the old baseline so dt
                # keeps accumulating to a meaningful window.
                MIN_DT = 0.05
                update_history = True
                if dt >= MIN_DT:
                    # Raw velocity from finite difference
                    raw_vel = (current_pos - prev_pos) / dt

                    # --- Adaptive EMA smoothing ---
                    # How different is the new raw reading from our smooth estimate?
                    # Large difference = direction change or acceleration = trust new reading more
                    # Small difference = consistent motion = trust history more
                    change = np.linalg.norm(raw_vel - prev_smooth)

                    # alpha scales from 0.2 (stable motion) to 0.9 (big change)
                    # change=0.0 → alpha=0.2 (80% history weight)
                    # change=2.3+ → alpha=0.9 (10% history, almost reset)
                    # EMA FIX: the adaptive alpha used to reach 0.9 on large
                    # changes — i.e. the MORE a reading disagreed with history,
                    # the MORE it was trusted. Good for real direction changes,
                    # but with +/-0.3m depth jitter at ~0.08s frames, a single
                    # noise spike looked like a huge "change" and was adopted
                    # almost raw (6+ m/s outputs from a 1 m/s walker). Capping
                    # alpha at 0.5 means no single frame can contribute more
                    # than half the estimate — spikes get halved, then decay,
                    # while genuine accelerations still come through in 2-3
                    # frames.
                    alpha = min(0.5, 0.2 + change * 0.3)

                    smooth_vel = alpha * raw_vel + (1.0 - alpha) * prev_smooth

                    velocity.x = float(smooth_vel[0])
                    velocity.y = float(smooth_vel[1])
                    velocity.z = float(smooth_vel[2])

                    self.get_logger().debug(
                        f" [{track_id}] raw=({raw_vel[0]:.2f},{raw_vel[1]:.2f}) "
                        f"change={change:.2f} alpha={alpha:.2f} "
                        f"smooth=({smooth_vel[0]:.2f},{smooth_vel[1]:.2f})")
                else:
                    # dt below MIN_DT (or negative): hold previous smooth AND
                    # publish it (the old code left the published velocity at
                    # zero here), and keep the old baseline for accumulation.
                    smooth_vel = prev_smooth
                    velocity.x = float(smooth_vel[0])
                    velocity.y = float(smooth_vel[1])
                    velocity.z = float(smooth_vel[2])
                    update_history = False

            else:
                # First time seeing this track — initialize smooth to zero
                smooth_vel = np.array([0.0, 0.0, 0.0])
                update_history = True

            # Update history: position, time, smoothed velocity.
            # When dt < MIN_DT the baseline is intentionally HELD (otherwise a
            # fast burst of frames resets it every cycle and dt never reaches
            # a differentiable window).
            if track_id not in self.track_history or update_history:
                self.track_history[track_id] = (current_pos, current_time, smooth_vel)

            speed = np.linalg.norm([velocity.x, velocity.y, velocity.z])
            movement_label = "moving" if speed > 0.05 else "static"

            tracked_person = TrackedPerson()
            tracked_person.name = str(track_id)
            tracked_person.position = Point(x=x, y=y, z=z)
            tracked_person.velocity = velocity
            tracked_person.reliability = 1.0
            tracked_person.tagnames = ["movement", "speed"]
            tracked_person.tags = [movement_label, f"{speed:.3f}"]
            self.pub_tracked_person.publish(tracked_person)
            tracked_people_msg.people.append(tracked_person)

            person = Person()
            person.name = f"track_{track_id}"
            person.position = tracked_person.position
            person.velocity = tracked_person.velocity
            person.reliability = tracked_person.reliability
            people_msg.people.append(person)

            self.get_logger().info(
                f"Published track_{track_id} | pos=({x:.2f}, {y:.2f}, {z:.2f}) "
                f"vel=({velocity.x:.2f}, {velocity.y:.2f}, {velocity.z:.2f}) | status={movement_label}"
            )

        # FIX 3: truthful frame id (world frame once TF has been seen,
        # source frame before that — either way, consumers transform correctly)
        frame_out = self.world_frame if self.cached_tf is not None else self.source_frame
        people_msg.header.frame_id = frame_out
        tracked_people_msg.header.frame_id = frame_out
        self.pub_tracked_people.publish(tracked_people_msg)
        self.pub_people.publish(people_msg)


def main(args=None):
    rclpy.init(args=args)
    node = VelocityTrackingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
