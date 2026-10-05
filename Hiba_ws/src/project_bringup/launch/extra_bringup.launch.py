def generate_launch_description():
    # Package Directory
    pkg_project_bringup = FindPackageShare('rplidar_ros')

    # Launch File
    device_launch = PathJoinSubstitution([pkg_project_bringup, 'launch', 'rplidar_a2m7_launch.py'])

    # Include Launch
    include_launch = IncludeLaunchDescription(PythonLaunchDescriptionSource([device_launch]))

    # Launch Description
    ld = LaunchDescription()
    ld.add_action(include_launch)
    return ld
