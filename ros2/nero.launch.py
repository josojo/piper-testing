"""NERO + gripper MoveIt stack. Hardware uses feedback, never GenericSystem."""
from pathlib import Path
import json
import tempfile
import xml.etree.ElementTree as ET

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def build(context):
    mode = LaunchConfiguration('mode').perform(context)
    namespace = LaunchConfiguration('namespace').perform(context).strip('/')
    if mode not in ('mock', 'hardware'):
        raise ValueError('mode must be mock or hardware')
    package = Path(get_package_share_directory('agx_arm_moveit'))
    mappings = {'arm_type': 'nero', 'effector_type': 'agx_gripper', 'revo2_type': 'left',
                'initial_positions_file': str(package / 'config/initial_positions.yaml')}
    config = (MoveItConfigsBuilder('agx_arm', package_name='agx_arm_moveit')
        .robot_description(file_path='config/agx_arm.urdf.xacro', mappings=mappings)
        .robot_description_semantic(file_path='config/agx_arm.srdf.xacro', mappings=mappings)
        .robot_description_kinematics(file_path='config/kinematics.yaml')
        .joint_limits(file_path='config/joint_limits.yaml')
        .trajectory_execution(file_path='config/moveit_controllers_none.yaml')
        .planning_pipelines(pipelines=['ompl'])
        .to_moveit_configs())
    # This experiment uses explicit collision boxes, not the vendor's example
    # point-cloud sensor. Do not auto-load its unavailable Octomap plugin.
    config.sensors_3d = {}
    joints = ['joint%d' % n for n in range(1, 8)]
    config.trajectory_execution['moveit_simple_controller_manager']['arm_controller']['joints'] = joints
    root = ET.fromstring(config.robot_description['robot_description'])
    hardware = root.find('./ros2_control/hardware')
    if hardware is None:
        raise RuntimeError('Vendor robot description has no ros2_control hardware')
    if mode == 'hardware':
        initial_file = LaunchConfiguration('initial_state_file').perform(context)
        if not initial_file:
            raise RuntimeError('Hardware launch requires a captured initial_state_file')
        initial = json.loads(Path(initial_file).read_text())
        values = dict(zip(joints, initial['joints_rad']))
        values['gripper'] = initial['gripper_width_m']
        for joint in root.findall('./ros2_control/joint'):
            initial_param = joint.find('./state_interface[@name="position"]/param[@name="initial_value"]')
            if initial_param is not None and joint.attrib['name'] in values:
                initial_param.text = str(values[joint.attrib['name']])
        hardware.find('plugin').text = 'topic_based_ros2_control/TopicBasedSystem'
        for name, value in {
            'joint_commands_topic': '/' + namespace + '/control/joint_commands',
            'joint_states_topic': '/' + namespace + '/feedback/joint_states',
            'trigger_joint_command_threshold': '-1',
        }.items():
            ET.SubElement(hardware, 'param', name=name).text = value
    # Model the attached gripper, but do not expose a gripper command controller.
    gripper = root.find('./ros2_control/joint[@name="gripper"]')
    for command in list(gripper.findall('command_interface')):
        gripper.remove(command)
    if mode == 'hardware':
        # TopicBasedSystem emits every joint name but only commanded positions.
        # A state-only gripper produces 8 names / 7 positions. Its measured state
        # already reaches MoveIt directly via feedback/joint_states; keep only
        # the seven commanded arm joints in this hardware transport.
        root.find('./ros2_control').remove(gripper)
    if mode == 'mock':
        gripper.find('./state_interface/param').text = '0.04'
    config.robot_description['robot_description'] = ET.tostring(root, encoding='unicode')
    for name in joints:
        config.joint_limits['robot_description_planning']['joint_limits'][name] = {
            'has_velocity_limits': True, 'max_velocity': 0.08,
            'has_acceleration_limits': True, 'max_acceleration': 0.15}
    # Action results must use actual hardware state; the vendor mock controller is
    # used only for mode=mock, in an isolated container network.
    controller_config = {
        '/' + namespace + '/controller_manager': {'ros__parameters': {
            'update_rate': 100,
            'arm_controller': {'type': 'joint_trajectory_controller/JointTrajectoryController'},
            'joint_state_broadcaster': {'type': 'joint_state_broadcaster/JointStateBroadcaster'}}},
        '/' + namespace + '/arm_controller': {'ros__parameters': {
            'joints': joints, 'command_interfaces': ['position'], 'state_interfaces': ['position', 'velocity'],
            'open_loop_control': False, 'allow_partial_joints_goal': False,
            'constraints': {'goal_time': 3.0, 'stopped_velocity_tolerance': 0.01,
                **{j: {'trajectory': 0.015, 'goal': 0.005} for j in joints}}}},
    }
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.safe_dump(controller_config, f)
        controller_path = f.name
    feedback = 'feedback/joint_states' if mode == 'hardware' else 'joint_states'
    return [
        Node(package='robot_state_publisher', executable='robot_state_publisher', namespace=namespace,
             parameters=[config.robot_description], remappings=[('joint_states', feedback)]),
        Node(package='moveit_ros_move_group', executable='move_group', namespace=namespace,
             parameters=[config.to_dict(), {'allow_trajectory_execution': True,
                'publish_robot_description_semantic': True, 'publish_planning_scene': True,
                'publish_geometry_updates': True, 'publish_state_updates': True,
                'publish_transforms_updates': True,
                'trajectory_execution.allowed_start_tolerance': 0.005}],
             remappings=[('joint_states', feedback)], output='screen'),
        Node(package='controller_manager', executable='ros2_control_node', namespace=namespace,
             parameters=[config.robot_description, controller_path], output='screen'),
        Node(package='controller_manager', executable='spawner', namespace=namespace,
             arguments=['joint_state_broadcaster', '--controller-manager', '/' + namespace + '/controller_manager',
                        '--controller-manager-timeout', '60']),
        Node(package='controller_manager', executable='spawner', namespace=namespace,
             arguments=['arm_controller', '--controller-manager', '/' + namespace + '/controller_manager',
                        '--controller-manager-timeout', '60']),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('mode', default_value='mock', choices=['mock', 'hardware']),
        DeclareLaunchArgument('namespace', default_value='nero'),
        DeclareLaunchArgument('initial_state_file', default_value=''),
        OpaqueFunction(function=build),
    ])
