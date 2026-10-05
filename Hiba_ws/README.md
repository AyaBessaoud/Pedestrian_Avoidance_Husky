# Master Thesis – Vision-based dynamic obstacle detection and tracking for mobile robots

## Focus of Work

- Integrate the YOLO-based dynamic obstacle detection algorithm into ROS 2 (Humble)
- Develop tracking logic to maintain consistent object identities across frames
- Display tracking results in RViz2
- Estimate the position of tracked persons using a depth camera
- Match tracked camera detections with corresponding 2D LiDAR-based obstacle positions
- Fuse LiDAR and depth data to enhance distance estimation
- Estimate position of tracked persons using fusion methods
- Integrate all components into a working navigation pipeline by adding a layer of navigation (`people_costmap_layer`) that takes input from the pose estimation node to enhance robot avoidance

---

## Folder Structure
```
SourceCode/
└── Src
    └── clearpath_ws/
        ├── clearpath_common                # Shared utilities and common code
        ├── clearpath_desktop               # Desktop tools and visualization configs
        ├── clearpath_robot                 # Robot-specific description and configuration
        ├── clearpath_simulator             # Simulation setup (Gazebo, Ignition)
        ├── fusion_tf                        # LiDAR–Camera TF fusion node
        ├── my_robot_launch                  # Launch files for robot bring-up
        ├── navigation                        # Navigation stack (localization, planning, control)
        ├── people                             # People detection/recognition
        ├── people_costmap_layer               # Custom costmap layer to avoid people
        ├── project_bringup                     # Bring-up package for full project integration
        ├── project_description                 # Robot and environment description (URDF, meshes)
        ├── realsense-ros                        # Intel RealSense camera driver
        ├── ros2_camera_lidar_fusion             # LiDAR projection onto camera images
        ├── rplidar_ros                           # RPLidar driver
        ├── tf_tree_frames                         # TF tree visualization and debugging tools
        ├── vision_tracking_pose                   # YOLO detection, pose estimation, and tracking
        └── vision_tracking_pose_msgs              # Custom messages for vision_tracking_pose
```

---

## Launch Instructions

### 1. Launch Simulation (Gazebo)
```bash
ros2 launch clearpath_gz simulation.launch.py \
  setup_path:=your_path/clearpath \
  namespace:=a200_0846 \
  use_sim_time:=true
```

### 2. Launch RViz2 with custom config
```bash
ros2 launch clearpath_viz view_robot.launch.py \
  config:=your_path/clearpath_ws/install/clearpath_viz/share/clearpath_viz/rviz/robot.rviz \
  use_sim_time:=true
```

### 3. Launch Localization
```bash
ros2 launch navigation localization.launch.py \
  namespace:=a200_0846 \
  use_sim_time:=true
```

---

## Approach 1: Depth Camera Based

#### Run Detection Node
```bash
ros2 run vision_tracking_pose detection
```

#### Run Pose Estimation Node (2D → 3D Conversion)
```bash
ros2 run vision_tracking_pose pose_estimation_depth
```

#### Run Tracking and Velocity Estimation Node
```bash
ros2 run vision_tracking_pose tracking_velocity
```

---

## Approach 2: LiDAR–Camera Fusion Based

#### Transform LiDAR Scan to Point Cloud
```bash
ros2 run fusion_tf scan_to_cloud
```

#### Run Detection and Tracking Node (Combined)
```bash
ros2 run vision_tracking_pose dete_track_sim
```

#### Run Fusion Node (Projection and Distance Estimation)
```bash
ros2 run fusion_tf fusion
```

---

### Manual Calibration Tools

Use the following nodes for manual calibration of LiDAR–camera projection:

| Node/Script                          | Description                                                       | Output                                     |
|--------------------------------------|-------------------------------------------------------------------|--------------------------------------------|
| `get_intrinsic_camera_calibration.py` | Computes intrinsic camera calibration.                             | Camera intrinsic calibration file          |
| `save_sensor_data.py`                 | Records synchronized LiDAR and camera data.                        | Sensor data file                            |
| `extract_points.py`                   | Manual selection of corresponding LiDAR–camera points.              | Corresponding points file                   |
| `get_extrinsic_camera_calibration.py` | Computes extrinsic LiDAR–camera calibration.                       | Extrinsic calibration file                  |
| `lidar_camera_projection`             | Projects LiDAR points into the camera frame for visualization.     | Projected points in image frame             |
| `projection.fusion.py`                | Estimates distance using LiDAR points inside detection bounding boxes. | Distance estimation results                 |

---

## Run Customized Navigation

The `people_costmap_layer` is integrated into the navigation stack to enhance dynamic obstacle avoidance. Launch the customized navigation with:

```bash
ros2 launch navigation nav2.launch.py \
  namespace:=a200_0846 \
  use_sim_time:=true
```

---

## Optional: Algorithm Comparison

To compare BoTSORT vs ByteTrack tracking algorithms:

```bash
ros2 run vision_tracking_pose test.py
```
*(Ensure `test.py` is executable: `chmod +x vision_tracking_pose/test.py`)*

---

## TF Visualization

To visualize TF frames:

```bash
ros2 run tf2_tools view_frames --ros-args \
--remap __node:=tf_viewer \
--remap /tf:=/a200_0846/tf \
--remap /tf_static:=/a200_0846/tf_static
```

---

## Notes

- Replace `your_path/clearpath` with the actual workspace path.  
- All nodes assume namespace `/a200_0846/`; adjust if using another robot name.  
- Always run `source install/setup.bash` after building your workspace.  
- Both approaches (Depth Camera and LiDAR–Camera Fusion) can be tested independently based on sensor availability.