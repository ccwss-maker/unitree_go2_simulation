from launch import LaunchDescription
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
from launch.actions import ExecuteProcess, TimerAction
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    package_share = FindPackageShare("gz_sim_launcher")

    # World file path
    world_path = PathJoinSubstitution([
        package_share,
        "worlds",
        "go2_plain.sdf"
    ])

    # Model path for GZ_SIM_RESOURCE_PATH
    model_path = PathJoinSubstitution([
        package_share,
        "models"
    ])

    # Topic bridge configuration
    topic_path = PathJoinSubstitution([
        package_share,
        "config",
        "go2_topic.yaml"
    ])

    pkg_path = get_package_share_directory("gz_sim_launcher")
    urdf_path_real = os.path.join(pkg_path, "models", "go2_description", "model.urdf")

    with open(urdf_path_real, 'r') as infp:
        robot_desc = infp.read()

    rviz_path = PathJoinSubstitution([
        package_share,
        "rviz",
        "go2.rviz"
    ])

    # Gazebo Sim with plugin path
    gz_sim = ExecuteProcess(
        cmd=["gz", "sim", world_path],
        output="screen",
        additional_env={
            "GZ_SIM_RESOURCE_PATH": model_path,
            "GZ_SIM_SYSTEM_PLUGIN_PATH": "/opt/ros/rolling/lib",
        }
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'robot_description': robot_desc},
        ]
    )

    # ROS-Gazebo bridge
    ros_gz_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        name="ros_gz_bridge",
        parameters=[{"config_file": topic_path}],
        output="screen"
    )

    # Odom to TF publisher
    odom_to_tf = Node(
        package='gz_sim_launcher',
        executable='odom_to_tf',
        name='odom_to_tf',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # Foot contact force processor
    foot_contact_processor = Node(
        package='gz_sim_launcher',
        executable='foot_contact_processor',
        name='foot_contact_processor',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    # Unitree legged control (PD + feedforward per joint)
    unitree_legged_control = Node(
        package='gz_sim_launcher',
        executable='unitree_legged_control',
        name='unitree_legged_control',
        output='screen',
        parameters=[{'use_sim_time': True}],
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_path],
        parameters=[{'use_sim_time': True}],
    )

    return LaunchDescription([
        gz_sim,
        robot_state_publisher,
        ros_gz_bridge,
        odom_to_tf,
        foot_contact_processor,
        unitree_legged_control,
        rviz,
    ])
