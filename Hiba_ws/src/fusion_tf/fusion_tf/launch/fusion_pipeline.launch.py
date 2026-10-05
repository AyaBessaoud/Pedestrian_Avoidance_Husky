from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    robot_ns = LaunchConfiguration('robot_ns')

    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_ns',
            default_value='a200_0846'
        ),

        Node(
            package='fusion_tf',
            executable='scan_to_cloud',
            name='scan_to_cloud',
            output='screen'
        ),

        Node(
            package='vision_tracking_pose',
            executable='dete_track_sim',
            name='dete_track_sim',
            output='screen'
        ),
         Node(
            package='vision_tracking_pose',
            executable='tracking_velocity',
            name='tracking_velocity',
            output='screen',
            remappings=[
                ('/tf', ['/', robot_ns, '/tf']),
                ('/tf_static', ['/', robot_ns, '/tf_static']),
            ]
        ),
        Node(
            package='vision_tracking_pose',
            executable='pose_estimation_depth',
            name='pose_estimation_depth',
            output='screen',
            parameters=[{'detection_topic': '/tracking', 'depth_image_units_divisor': 1.0}],
            remappings=[
                ('/tf', ['/', robot_ns, '/tf']),
                ('/tf_static', ['/', robot_ns, '/tf_static']),
            ]
        ),
        Node(
            package='fusion_tf',
            executable='fusion',
            name='fusion',
            output='screen',
            remappings=[
                ('/tf', ['/', robot_ns, '/tf']),
                ('/tf_static', ['/', robot_ns, '/tf_static']),
            ]
        ),
    ])
