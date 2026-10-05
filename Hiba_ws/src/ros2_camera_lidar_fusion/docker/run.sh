# For graphics

isRunning=`docker ps -f name=v | grep -c "ros2_camera_lidar_fusion"`;

if [ $isRunning -eq 0 ]; then
    xhost +local:docker
    docker rm ros2_camera_lidar_fusion
    docker run \
        --name ros2_camera_lidar_fusion \
        -it \
        --env="DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
        --net host \
        --ipc host \
        --pid host \
        --privileged \
        --volume `pwd`/../:/ros2_ws/src/ros2_camera_lidar_fusion \
        -w /ros2_ws \
        ros2_camera_lidar_fusion:latest

else
    echo "ros2_camera_lidar_fusion is already running"
    docker exec -it ros2_camera_lidar_fusion /bin/bash
fi
