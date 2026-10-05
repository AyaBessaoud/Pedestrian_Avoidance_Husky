# Pedestrian-Aware Navigation for Mobile Robots Using NMPC

Graduation project (PFE), INSAT Tunis, carried out at the ADS Lab, TU Dresden (DAAD KOSPIE).

A Nonlinear Model Predictive Control (NMPC) layer is added on top of the standard ROS 2 Nav2 stack of a Clearpath Husky A200. Nav2 (DWB) drives the robot normally. When a pedestrian is predicted to cross the robot's path, the NMPC node takes control through `twist_mux`, performs a give-way maneuver toward a target point beyond the crossing, then hands control back to Nav2.

- **Robot:** Clearpath Husky A200 (namespace `a200_0846`)
- **Software:** ROS 2 Humble, Nav2, Gazebo, CasADi + IPOPT, Optuna (TPE)
- **Sensors:** Intel RealSense D435 (RGB-D) + RPLIDAR (2D LiDAR)

### How it works

```
/people ──▶ threat check: is the person moving, close, and crossing the path ahead?
                 │ yes
                 ▼
         give-way target ("carrot"): predict the person 2 s ahead,
         pick a path point beyond the predicted crossing
                 │
                 ▼
         NMPC control loop @ 10 Hz (CasADi/IPOPT, N = 30, dt = 0.1 s)
         drive to the carrot while avoiding LiDAR obstacles
                 │ carrot reached
                 ▼
         NMPC deactivates → twist_mux → Nav2/DWB resumes
```

ROS 2 runs the **installed** copy of the node (after `colcon build`), not the source file, so rebuild after every change.

---

## 1. Repository structure

```
Husky/
├── Hiba_ws/src/
│   ├── my_nmpc_controller/          # NMPC node (nmpc_node.py) + weights (config/)
│   └── ...                          # perception, fusion and robot description packages
├── warehouse.sdf                    # Gazebo world with the walking pedestrian (actor)
├── nmpc_optimizer_final.py          # weight tuning (Bayesian optimization)
├── simple_metrics_node.py           # evaluation metrics recorder
├── person_lidar_carryover_node.py   # keeps tracking a person with the LiDAR
├── lidar_camera_overlay.py          # real robot: check camera/LiDAR alignment
└── README.md
```

Not included: `clearpath/` and `clearpath_ws/` (Clearpath robot configuration and simulator packages, set up separately).

---

## 2. Build

```bash
cd Hiba_ws
colcon build
source install/setup.bash
```

Every terminal needs these sources (they are not in `~/.bashrc`, so you can choose the workspace per terminal):

```bash
source /opt/ros/humble/setup.bash
source ~/Downloads/Husky/clearpath_ws/install/setup.bash
source ~/Downloads/Husky/Hiba_ws/install/setup.bash
```

> Note: the scripts use the path `~/Downloads/Husky/`. Change it if your workspace is somewhere else.

---

## 3. Simulation world and pedestrian

The pedestrian is a Gazebo **actor** (`actor_walking3`) in `warehouse.sdf`. It walks a fixed path in a loop, by itself, independent of the robot.

The file contains **two paths**: one is active and the other is commented out. Swap them depending on the task:

| Task | Active pedestrian path |
|---|---|
| Weight tuning (section 5) | vertical pacer: x = 2.0, between y = -0.5 and y = 3.0 |
| Evaluation (section 6) | between (7.5, 1.6) and (2.0, 0.6) |

Using a different path for tuning and for evaluation keeps the evaluation fair (the controller is not tested on the scenario it was tuned on).

Copy the world file to where the Clearpath simulator loads it (both the `src` and the `install` copies), then restart Gazebo completely:

```bash
cp warehouse.sdf ~/Downloads/Husky/clearpath_ws/src/clearpath_simulator/clearpath_gz/worlds/
cp warehouse.sdf ~/Downloads/Husky/clearpath_ws/install/clearpath_gz/share/clearpath_gz/worlds/
```

---

## 4. Simulation stack (Gazebo)

Start these in order, each in its own terminal, and keep them running:

```bash
# T1: RViz
ros2 launch clearpath_viz view_navigation.launch.py namespace:=a200_0846 use_sim_time:=true

# T2: Gazebo simulation
ros2 launch clearpath_gz simulation.launch.py setup_path:=$HOME/Downloads/Husky/clearpath/

# T3: Localization (then set the 2D Pose Estimate in RViz)
ros2 launch clearpath_nav2_demos localization.launch.py setup_path:=$HOME/Downloads/Husky/clearpath/ use_sim_time:=true

# T4: Simulated LiDAR bridge
ros2 launch my_lidar_sim lidar_sim_bridge.launch.py

# T5: Perception (person detection, tracking, 3D position)
ros2 launch fusion_tf fusion_pipeline.launch.py

# T6: Nav2
ros2 launch clearpath_nav2_demos nav2.launch.py setup_path:=$HOME/Downloads/Husky/clearpath/ use_sim_time:=true

# T7: tf_static relay (launch once)
ros2 run topic_tools relay /a200_0846/tf_static /tf_static
```

Wait for Nav2 to print `Managed nodes are active`.

---

## 5. Weight tuning (Bayesian optimization)

The 7 NMPC cost weights are tuned with Optuna (TPE sampler):

- **Trial 1** uses the **Bryson's rule** baseline (w_i = 1 / max_acceptable_i²).
- **150 trials**. Each trial launches the NMPC node with new weights, drives 2 goals through a pacing pedestrian, and is scored:
  `score = 0.4·safety + 0.3·smoothness + 0.2·success + 0.1·time`
- If NMPC never activates in a trial, the same weights are retried up to 2 times (activation depends on perception timing, not on the weights).

Set the world file to the **tuning path** (section 3). With the simulation stack (section 4) running:

```bash
python3 nmpc_optimizer_final.py
```

- Results: `~/nmpc_optimization_results.json`
- Best weights: `Hiba_ws/src/my_nmpc_controller/config/best_weights.yaml` (created by the optimizer)

---

## 6. Evaluation (NMPC vs DWB)

Set the world file to the **evaluation path** (section 3). The pedestrian walks the same path, and the robot starts from the same position, in both scenarios. **Only the goal changes:**

- **Perpendicular crossing:** a goal whose route crosses the pedestrian's path at about a right angle.
- **Head-on approach:** a goal whose route goes toward the pedestrian, so the person walks toward the robot.

Run 10 trials per controller (NMPC and DWB) in each scenario.

> Note: the social costmap layer (`people_costmap_layer`) was disabled during testing. When it is on, DWB reacts to the person before NMPC can activate, which hides the behavior being tested.

**Start (in addition to section 4):**

```bash
# LiDAR carryover: keeps tracking the person with the LiDAR when the camera loses them
python3 person_lidar_carryover_node.py

# Metrics recorder (runs in the background)
python3 simple_metrics_node.py
```

**NMPC trials only**, also start the NMPC node (it reads the extended people topic):

```bash
ros2 run my_nmpc_controller nmpc_node --ros-args \
  -r /tf:=/a200_0846/tf -r /tf_static:=/a200_0846/tf_static \
  -r /people:=/people_lidar_extended
```

The node uses the default weights written in `nmpc_node.py`. To run with weights from your own tuning run instead, add `--params-file Hiba_ws/src/my_nmpc_controller/config/best_weights.yaml`.

**DWB trials:** do not start the NMPC node; Nav2 drives alone.

**Each trial:** set **2D Pose Estimate** in RViz once at the start, then send the scenario's goal with **2D Goal Pose**. The metrics node starts and stops the trial by itself, using Nav2's goal status (accepted → start; succeeded / aborted / canceled → end).

**Metrics** (saved to `~/simple_metrics_results.json`, one entry per trial):

| Metric | How it is measured |
|---|---|
| Minimum pedestrian distance | robot ↔ tracked person (camera-only and camera + LiDAR) |
| Success | Nav2 reached the goal and no collision occurred |
| Collisions | closest raw LiDAR return, corrected to `base_link` |
| Time to goal | from Nav2 goal accepted to goal finished |
| Maximum path deviation | largest distance from the original Nav2 path |
| Command smoothness | average linear / angular jerk of the executed `cmd_vel` |

---

## 7. Real robot

### Connect

```bash
ssh administrator@192.168.0.199          # over Wi-Fi (hostname cpr-a200-0846)
sudo systemctl restart clearpath-robot.service
ros2 topic echo /a200_0846/platform/emergency_stop --once   # check the e-stop
```

### Ethernet link (needed for the camera)

Wi-Fi drops most camera frames, so the laptop must use the robot's wired network (`192.168.131.x`).

```bash
ip addr                                   # find your enx... interface (LOWER_UP)
sudo ip addr add 192.168.131.100/24 dev <your_enx_interface>
sudo ip link set <your_enx_interface> up
ping -c 5 192.168.131.1
ssh administrator@192.168.131.1
```

In **every** terminal (robot and laptop):

```bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
ros2 topic list                           # should show the /a200_0846/... topics
```

### Sensors

```bash
lsusb | grep -i intel                     # camera detected
ros2 pkg list | grep realsense            # realsense2_camera, _msgs, _description
ros2 topic hz /a200_0846/sensors/camera_0/color/image

# LiDAR (on the robot, launched manually)
ros2 launch rplidar_ros rplidar_a2m12_launch.py serial_port:=/dev/rplidar
```

### Perception (on the laptop, the robot has no GPU)

```bash
ros2 launch fusion_tf fusion_pipeline.launch.py
ros2 topic echo /tracking                 # 2D detections
ros2 topic echo /detections_3d --once     # 3D positions
```

### Optional: check camera / LiDAR alignment

```bash
python3 lidar_camera_overlay.py
ros2 run rqt_image_view rqt_image_view    # view /a200_0846/lidar_camera_overlay
```

> Safety: keep the e-stop within reach, and test small movements first.

---

## 8. Credits

The perception pipeline (person detection, tracking and fusion) and the robot description packages were developed by colleagues at the ADS Lab, TU Dresden. Third-party packages (`realsense-ros`, `rplidar_ros`, Clearpath packages) belong to their authors.
