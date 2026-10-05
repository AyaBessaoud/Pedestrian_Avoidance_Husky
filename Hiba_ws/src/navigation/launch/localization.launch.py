from ament_index_python.packages import get_package_share_directory
from clearpath_config.common.utils.yaml import read_yaml
from clearpath_config.clearpath_config import ClearpathConfig
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import EnvironmentVariable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import PushRosNamespace, Node

ARGUMENTS = [
    DeclareLaunchArgument('use_sim_time', default_value='true', choices=['true', 'false'], description='Use sim time'),
    DeclareLaunchArgument('setup_path', default_value=[EnvironmentVariable('HOME'), '/clearpath/'], description='Clearpath setup path'),
    DeclareLaunchArgument('namespace', default_value='a200_0846', description='Namespace for the robot')
]

def launch_setup(context, *args, **kwargs):
    pkg_clearpath_nav2_demos = get_package_share_directory('navigation')
    pkg_nav2_bringup = get_package_share_directory('nav2_bringup')

    use_sim_time = LaunchConfiguration('use_sim_time')
    setup_path = LaunchConfiguration('setup_path')
    namespace = LaunchConfiguration('namespace')
    map = LaunchConfiguration('map')

    # Read robot config
    config = read_yaml(setup_path.perform(context) + 'robot.yaml')
    clearpath_config = ClearpathConfig(config)
    platform_model = clearpath_config.platform.get_platform_model()

    # Localization paths
    file_parameters = PathJoinSubstitution([pkg_clearpath_nav2_demos, 'config', platform_model, 'localization.yaml'])
    launch_localization = PathJoinSubstitution([pkg_nav2_bringup, 'launch', 'localization_launch.py'])

    # Map server as a standard node with a small delay
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        namespace=namespace,
        name='map_server',
        output='screen',
        parameters=[{'yaml_filename': '/home/hiba/clearpath_ws/install/navigation/share/navigation/maps/warehouse.yaml'}, {'use_sim_time': True}]
    )

    delayed_map_server = TimerAction(
        period=3.0,
        actions=[map_server_node]
    )

    # AMCL as a standard node
    amcl_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(launch_localization),
        launch_arguments=[
            ('namespace', namespace),
            ('map', map),
            ('use_sim_time', use_sim_time),
            ('params_file', file_parameters)
        ]
    )

    # Localization group
    localization = GroupAction([
        PushRosNamespace(namespace),
        delayed_map_server,
        amcl_node
    ])

    return [localization]

def generate_launch_description():
    pkg_clearpath_nav2_demos = get_package_share_directory('navigation')
    map_arg = DeclareLaunchArgument(
        'map',
        default_value=PathJoinSubstitution([pkg_clearpath_nav2_demos, 'maps', 'warehouse.yaml']),
        description='Full path to map yaml file to load'
    )

    ld = LaunchDescription(ARGUMENTS)
    ld.add_action(map_arg)
    ld.add_action(OpaqueFunction(function=launch_setup))
    return ld

