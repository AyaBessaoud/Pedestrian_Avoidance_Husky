#!/usr/bin/env python3
import subprocess
import time
import json
import os
import signal
import threading
import optuna
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav2_msgs.action import NavigateToPose
from people_msgs.msg import People
import numpy as np
from datetime import datetime

N_TRIALS = 150
RESULTS_FILE = os.path.expanduser("~/nmpc_optimization_results.json")
MAX_INVALID_RETRIES = 2
SETUP_WAIT = 15.0
GOAL_TIMEOUT = 120.0
RETURN_PROXIMITY = 1.5
ROBOT_START_X = 0.0
ROBOT_START_Y = 0.0
ROBOT_START_YAW = 0.0
GOALS = [(8.5, 1.5, 0.0), (-0.5, 1.5, 3.14), (0.0, 0.0, 0.0)]
SOURCE_CMD = "source /opt/ros/humble/setup.bash && source ~/Downloads/Husky/clearpath_ws/install/setup.bash && source ~/Downloads/Husky/Hiba_ws/install/setup.bash"
WEIGHT_BOUNDS = {
    "w_pos": (0.5, 110.0),
    "w_heading": (0.1, 10.0),
    "w_vel": (0.01, 1.0),
    "w_smooth": (0.1, 30.0),
    "w_obs": (4.5, 60.0),
    "w_terminal_pos": (5.0, 350.0),
    "w_terminal_h": (0.5, 20.0),
}
BASELINE = {
    "w_pos": 1.0,
    "w_heading": 3.6476,
    "w_vel": 1.0,
    "w_smooth": 25.0,
    "w_obs": 4.9383,
    "w_terminal_pos": 6.25,
    "w_terminal_h": 14.5903,
}


class MetricsCollector(Node):

    def __init__(self):
        super().__init__("nmpc_metrics_collector")
        self.reset()
        self.create_subscription(
            PoseWithCovarianceStamped, "/a200_0846/amcl_pose", self._pose_cb, 10
        )
        self.create_subscription(People, "/people", self._people_cb, 10)
        self.create_subscription(Twist, "/a200_0846/nmpc/cmd_vel", self._cmdvel_cb, 10)
        self._nav_client = ActionClient(self, NavigateToPose, "/a200_0846/navigate_to_pose")
        self._current_goal_id = 0
        self._expected_goal_id = 0
        self.get_logger().info("[METRICS] Collector ready")

    def send_nav_goal(self, x, y, yaw):
        if not self._nav_client.wait_for_server(timeout_sec=8.0):
            print(f"  [GOAL] Action server not available!")
            return False
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = "map"
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.position.z = 0.0
        goal.pose.pose.orientation.z = float(np.sin(yaw / 2))
        goal.pose.pose.orientation.w = float(np.cos(yaw / 2))
        self._current_goal_id += 1
        self._expected_goal_id = self._current_goal_id
        self._goal_event.clear()
        self._goal_succeeded = False
        future = self._nav_client.send_goal_async(goal)
        future.add_done_callback(
            lambda f, gid=self._current_goal_id: self._goal_response_cb(f, gid)
        )
        print(f"  [GOAL] Sent → ({x:.1f}, {y:.1f}) [id={self._current_goal_id}]")
        return True

    def _goal_response_cb(self, future, goal_id):
        if goal_id != self._expected_goal_id:
            print(
                f"  [STALE] Ignoring old goal response (id={goal_id}, expected={self._expected_goal_id})"
            )
            return
        handle = future.result()
        if not handle.accepted:
            print("  [GOAL] Rejected by Nav2!")
            self._goal_event.set()
            return
        handle.get_result_async().add_done_callback(
            lambda f, gid=goal_id: self._goal_result_cb(f, gid)
        )

    def _goal_result_cb(self, future, goal_id):
        if goal_id != self._expected_goal_id:
            print(
                f"  [STALE] Ignoring old goal result (id={goal_id}, expected={self._expected_goal_id})"
            )
            return
        status = future.result().status
        self._goal_succeeded = status == 4
        self._goal_event.set()

    def invalidate_current_goal(self):
        self._expected_goal_id = -1
        self._goal_event.set()

    def reset(self):
        self.min_person_dist = float("inf")
        self.velocity_changes = []
        self.last_pose = None
        self.person_pos = None
        self.goals_reached = 0
        self.start_time = time.time()
        self.active = True
        self.last_v = None
        self.last_w = None
        self.nmpc_activations = 0
        if not hasattr(self, "_goal_event"):
            self._goal_event = threading.Event()
        self._goal_event.clear()
        self._goal_succeeded = False
        if hasattr(self, "_expected_goal_id"):
            self._expected_goal_id = -1

    def _cmdvel_cb(self, msg):
        if not self.active:
            return
        v = msg.linear.x
        w = msg.angular.z
        if self.last_v is not None:
            dv = abs(v - self.last_v) + abs(w - self.last_w)
            self.velocity_changes.append(dv)
            if abs(v) > 0.01 or abs(w) > 0.01:
                self.nmpc_activations += 1
        self.last_v = v
        self.last_w = w

    def _pose_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.last_pose = (x, y)
        if self.person_pos is not None:
            d = np.hypot(x - self.person_pos[0], y - self.person_pos[1])
            self.min_person_dist = min(self.min_person_dist, d)

    def _people_cb(self, msg):
        if not self.active or len(msg.people) == 0:
            return
        p = msg.people[0]
        self.person_pos = (p.position.x, p.position.y)
        if self.last_pose is not None:
            d = np.hypot(
                self.last_pose[0] - self.person_pos[0], self.last_pose[1] - self.person_pos[1]
            )
            self.min_person_dist = min(self.min_person_dist, d)

    def compute_score(self):
        if self.nmpc_activations == 0:
            print("  [SCORE] INVALID — NMPC never activated, score=0.0")
            return (
                0.0,
                {
                    "invalid": True,
                    "reason": "nmpc_activations=0",
                    "nmpc_activations": 0,
                    "goals_reached": self.goals_reached,
                    "min_person_dist": (
                        round(self.min_person_dist, 3)
                        if self.min_person_dist < float("inf")
                        else -1
                    ),
                },
            )
        PEDESTRIAN_GOALS = len(GOALS) - 1
        safety = min(self.min_person_dist, 3.0) / 3.0
        if self.min_person_dist == float("inf"):
            safety = 0.3
        jerk = float(np.mean(self.velocity_changes)) if len(self.velocity_changes) >= 2 else 999.0
        smoothness = 1.0 / (1.0 + jerk * 10.0)
        success = min(self.goals_reached, PEDESTRIAN_GOALS) / PEDESTRIAN_GOALS
        elapsed = time.time() - self.start_time
        time_score = 1.0 - min(elapsed / (GOAL_TIMEOUT * PEDESTRIAN_GOALS), 1.0)
        score = 0.4 * safety + 0.3 * smoothness + 0.2 * success + 0.1 * time_score
        return (
            score,
            {
                "safety": round(safety, 3),
                "smoothness": round(smoothness, 3),
                "jerk": round(jerk, 4),
                "goals_reached": self.goals_reached,
                "total_goals": len(GOALS),
                "success": round(success, 2),
                "time": round(elapsed, 1),
                "min_person_dist": (
                    round(self.min_person_dist, 3) if self.min_person_dist < float("inf") else -1
                ),
                "nmpc_activations": self.nmpc_activations,
            },
        )


class ProcessManager:

    def __init__(self):
        self.processes = []

    def launch(self, cmd, name="node"):
        p = subprocess.Popen(
            f"bash -c '{SOURCE_CMD} && {cmd}'",
            shell=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            preexec_fn=os.setsid,
        )
        self.processes.append((name, p))
        print(f"  [LAUNCH] {name} (pid={p.pid})")
        return p

    def kill_all(self):
        for name, p in self.processes:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                print(f"  [KILL] {name}")
            except Exception:
                pass
        self.processes = []
        time.sleep(3.0)
        subprocess.run("pkill -f 'nmpc_node' 2>/dev/null", shell=True)
        time.sleep(3.0)


def write_weights(w):
    path = os.path.expanduser(
        "~/Downloads/Husky/Hiba_ws/src/my_nmpc_controller/config/weights.yaml"
    )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("husky_nmpc_node:\n  ros__parameters:\n")
        for k, v in w.items():
            f.write(f"    {k}: {v:.4f}\n")
    print(f"  [CONFIG] Weights written to {path}")
    return path


def execute_goal(metrics, gx, gy, gyaw, label, is_return=False, max_attempts=3):
    for attempt in range(max_attempts):
        if attempt > 0:
            print(f"  [RETRY] Attempt {attempt + 1}/{max_attempts} for {label}")
            time.sleep(3.0)
        sent = metrics.send_nav_goal(gx, gy, gyaw)
        if not sent:
            print(f"  [GOAL] Failed to send — action server unavailable")
            continue
        time.sleep(5.0)
        goal_start = time.time()
        last_log = -1
        last_pos = None
        stuck_count = 0
        while not metrics._goal_event.wait(timeout=0.5):
            elapsed = time.time() - goal_start
            if is_return and metrics.last_pose is not None:
                d = np.hypot(metrics.last_pose[0] - gx, metrics.last_pose[1] - gy)
                if d < RETURN_PROXIMITY:
                    print(f"  [RETURN] Close enough (dist={d:.2f}m < {RETURN_PROXIMITY}m) — done")
                    metrics.invalidate_current_goal()
                    return True
            if elapsed > GOAL_TIMEOUT:
                if metrics.last_pose is not None:
                    d = np.hypot(metrics.last_pose[0] - gx, metrics.last_pose[1] - gy)
                    if d < 1.5:
                        print(f"  [CLOSE ENOUGH] {label} dist={d:.2f}m — counting as reached")
                        metrics.invalidate_current_goal()
                        return True
                print(f"  ⚠ {label} timed out after {GOAL_TIMEOUT:.0f}s")
                metrics.invalidate_current_goal()
                break
            if metrics.last_pose is not None:
                if last_pos is not None:
                    moved = np.hypot(
                        metrics.last_pose[0] - last_pos[0], metrics.last_pose[1] - last_pos[1]
                    )
                    stuck_count = 0 if moved > 0.05 else stuck_count + 1
                last_pos = metrics.last_pose
                if stuck_count > 120:
                    print(f"  ⚠ {label} — robot stuck 60s, retrying")
                    metrics.invalidate_current_goal()
                    break
                if int(elapsed) % 10 == 0 and int(elapsed) != last_log:
                    last_log = int(elapsed)
                    d = np.hypot(metrics.last_pose[0] - gx, metrics.last_pose[1] - gy)
                    activated = "NMPC" if metrics.nmpc_activations > 0 else "Nav2"
                    print(
                        f"  t={elapsed:.0f}s | robot=({metrics.last_pose[0]:.1f},{metrics.last_pose[1]:.1f}) | dist={d:.2f}m | person={metrics.min_person_dist:.2f}m | {activated}({metrics.nmpc_activations})"
                    )
            elif int(elapsed) % 10 == 0 and int(elapsed) != last_log:
                last_log = int(elapsed)
                print(f"  t={elapsed:.0f}s | robot=unknown")
        if metrics._goal_succeeded:
            elapsed = time.time() - goal_start
            print(f"  ✓ {label} reached (t={elapsed:.1f}s)")
            return True
    print(f"  ✗ {label} failed after {max_attempts} attempts — skipping")
    return False


def run_trial(weights, trial_num, metrics, executor):
    print(f"\n{'=' * 60}\nTRIAL {trial_num}")
    for k, v in weights.items():
        print(f"  {k}={v:.3f}", end="")
    print(f"\n{'=' * 60}")
    config_path = write_weights(weights)
    pm = ProcessManager()
    try:
        print("\n[SETUP] Launching NMPC...")
        pm.launch(
            f"ros2 run my_nmpc_controller nmpc_node --ros-args -r /tf:=/a200_0846/tf -r /tf_static:=/a200_0846/tf_static --params-file {config_path}",
            name="nmpc",
        )
        time.sleep(3.0)
        print(f"[SETUP] Waiting {SETUP_WAIT}s for NMPC + Nav2 to stabilize...")
        time.sleep(SETUP_WAIT)
        metrics.reset()
        time.sleep(1.0)
        PEDESTRIAN_GOALS = len(GOALS) - 1
        for goal_idx, (gx, gy, gyaw) in enumerate(GOALS):
            is_return = goal_idx == len(GOALS) - 1
            label = "RETURN" if is_return else f"GOAL {goal_idx + 1}/{PEDESTRIAN_GOALS}"
            print(f"\n[{label}] → ({gx:.1f}, {gy:.1f})")
            if is_return:
                metrics.active = False
            if is_return and metrics.last_pose is not None:
                d = np.hypot(metrics.last_pose[0] - gx, metrics.last_pose[1] - gy)
                if d < RETURN_PROXIMITY:
                    print(f"  [RETURN] Already at origin (dist={d:.2f}m) — skipping")
                    continue
            reached = execute_goal(
                metrics, gx, gy, gyaw, label, is_return=is_return, max_attempts=3
            )
            if reached and (not is_return):
                metrics.goals_reached += 1
            if not is_return:
                time.sleep(2.0)
        metrics.active = False
        score, metrics_dict = metrics.compute_score()
        metrics_dict["score"] = round(score, 4)
        print(f"\n[RESULT] Trial {trial_num}:")
        if metrics_dict.get("invalid"):
            print(f"  INVALID — {metrics_dict.get('reason')}")
        else:
            print(
                f"  safety={metrics_dict['safety']:.3f} | smoothness={metrics_dict['smoothness']:.3f} | jerk={metrics_dict['jerk']:.4f}"
            )
            print(
                f"  goals={metrics_dict['goals_reached']}/{metrics_dict['total_goals']} | min_person_dist={metrics_dict['min_person_dist']:.3f}m | nmpc_activations={metrics_dict['nmpc_activations']}"
            )
        print(f"  SCORE = {score:.4f}")
        return (score, metrics_dict)
    except Exception as e:
        print(f"[ERROR] Trial {trial_num} failed: {e}")
        import traceback

        traceback.print_exc()
        return (0.0, {"error": str(e)})
    finally:
        pm.kill_all()
        time.sleep(2.0)


trial_results = []
_metrics = None
_executor = None


def objective(trial):
    weights = {k: trial.suggest_float(k, lo, hi) for k, (lo, hi) in WEIGHT_BOUNDS.items()}
    trial_num = len(trial_results) + 1
    attempts_used = 0
    for attempt in range(MAX_INVALID_RETRIES + 1):
        attempts_used = attempt + 1
        label = str(trial_num) if attempt == 0 else f"{trial_num}-retry{attempt}"
        score, metrics_dict = run_trial(weights, label, _metrics, _executor)
        if not metrics_dict.get("invalid"):
            break
        if attempt < MAX_INVALID_RETRIES:
            print(
                f"\n[RETRY] Trial {trial_num}: NMPC never activated (perception/timing, not the weights) -- retrying same weights, attempt {attempt + 2}/{MAX_INVALID_RETRIES + 1}"
            )
    metrics_dict["retry_attempts"] = attempts_used
    result = {
        "trial": trial_num,
        "weights": {k: round(v, 4) for k, v in weights.items()},
        "metrics": metrics_dict,
        "score": score,
        "time": datetime.now().isoformat(),
    }
    trial_results.append(result)
    with open(RESULTS_FILE, "w") as f:
        json.dump(trial_results, f, indent=2)
    valid_results = [r for r in trial_results if r["score"] > 0]
    invalid_results = [r for r in trial_results if r["score"] == 0]
    best = max((r["score"] for r in valid_results), default=0.0)
    print(
        f"\n[OPTUNA] Trial {trial_num} score={score:.4f} (attempts used: {attempts_used}/{MAX_INVALID_RETRIES + 1} | best valid: {best:.4f} | valid: {len(valid_results)}/{trial_num} | invalid after retries: {len(invalid_results)}/{trial_num})"
    )
    return score


def main():
    print("=" * 60)
    print("NMPC BAYESIAN WEIGHT OPTIMIZER (Bryson's rule baseline)")
    print("=" * 60)
    print(f"Trials:          {N_TRIALS}")
    print(f"Results file:    {RESULTS_FILE}")
    print(f"Pedestrian zone: x=2.0, y=-0.5 to y=3.0")
    print(f"Goals per trial: {len(GOALS)}")
    for i, (x, y, _) in enumerate(GOALS):
        print(f"  Goal {i + 1}: ({x:.1f}, {y:.1f})")
    print()
    print("START THESE MANUALLY and keep running for ALL trials:")
    print("  T1: ros2 launch clearpath_viz view_navigation.launch.py")
    print("        namespace:=a200_0846 use_sim_time:=true")
    print("  T2: ros2 launch clearpath_gz simulation.launch.py")
    print("        setup_path:=$HOME/Downloads/Husky/clearpath/")
    print("  T3: ros2 launch clearpath_nav2_demos localization.launch.py")
    print("        setup_path:=$HOME/Downloads/Husky/clearpath/ use_sim_time:=true")
    print("  T4: ros2 launch my_lidar_sim lidar_sim_bridge.launch.py")
    print("  T5: ros2 launch fusion_tf fusion_pipeline.launch.py")
    print("  T6: ros2 launch clearpath_nav2_demos nav2.launch.py")
    print("        setup_path:=$HOME/Downloads/Husky/clearpath/ use_sim_time:=true")
    print("  T7: ros2 run topic_tools relay /a200_0846/tf_static /tf_static")
    print("      ← launch once, leave running")
    print()
    print("Wait for Nav2 'Managed nodes are active', then press ENTER.")
    print()
    input("Press ENTER when all terminals are ready...")
    global _metrics, _executor
    rclpy.init()
    _metrics = MetricsCollector()
    _executor = rclpy.executors.SingleThreadedExecutor()
    _executor.add_node(_metrics)

    def spin_forever():
        while rclpy.ok():
            try:
                _executor.spin_once(timeout_sec=0.1)
            except Exception:
                pass

    threading.Thread(target=spin_forever, daemon=True).start()
    time.sleep(2.0)
    study = optuna.create_study(
        direction="maximize",
        study_name="nmpc_weight_optimization",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.enqueue_trial(BASELINE)
    print("[INFO] Trial 1 = Bryson's rule baseline weights")
    print(
        f"[INFO] Trials with nmpc_activations=0 are retried up to {MAX_INVALID_RETRIES} times, then score 0.0"
    )
    print()
    study.optimize(objective, n_trials=N_TRIALS)
    best = study.best_trial
    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best score: {best.value:.4f}")
    print("Best weights:")
    for k, v in best.params.items():
        pct = (v - BASELINE[k]) / BASELINE[k] * 100
        print(f"  {k:20s}: {v:.4f}  (Bryson baseline was {BASELINE[k]:.4f}, {pct:+.1f}%)")
    write_weights(best.params)
    best_path = os.path.expanduser(
        "~/Downloads/Husky/Hiba_ws/src/my_nmpc_controller/config/best_weights.yaml"
    )
    import shutil

    shutil.copy(
        os.path.expanduser("~/Downloads/Husky/Hiba_ws/src/my_nmpc_controller/config/weights.yaml"),
        best_path,
    )
    valid = [r for r in trial_results if r["score"] > 0]
    invalid = [r for r in trial_results if r["score"] == 0]
    print(f"\nValid trials (NMPC activated): {len(valid)}/{len(trial_results)}")
    print(f"Invalid trials (NMPC silent):  {len(invalid)}/{len(trial_results)}")
    if invalid:
        print("Invalid trial numbers:", [r["trial"] for r in invalid])
    print(f"\nBest weights saved to: {best_path}")
    print(f"Full results at:       {RESULTS_FILE}")
    print()
    print("To use best weights:")
    print(f"  ros2 run my_nmpc_controller nmpc_node --ros-args \\")
    print(f"    -r /tf:=/a200_0846/tf -r /tf_static:=/a200_0846/tf_static \\")
    print(f"    --params-file {best_path}")


if __name__ == "__main__":
    main()
