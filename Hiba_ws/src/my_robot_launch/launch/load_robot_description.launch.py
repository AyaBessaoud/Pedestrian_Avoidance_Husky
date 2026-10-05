from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import Command, FindExecutable

def generate_launch_description():
    xacro_path = '/home/hiba/clearpath_sim/robot.urdf.xacro'

    return LaunchDescription([
        # Robot State Publisher
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': Command([
                    FindExecutable(name='xacro'), ' ', xacro_path
                ]),
                'use_sim_time': True
            }]
        ),
    ])

