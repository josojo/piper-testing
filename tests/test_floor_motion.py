"""Floor-profile regression checks: offline models and fake hardware only."""
from dataclasses import replace
import json
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

# Load before patch.dict(sys.modules) so ROS mocks cannot orphan this module.
import nero_agent.abort_policy

from nero_agent.core import AgentError, Settings, JOINTS, FLOOR_EXCURSION_RAD, FLOOR_MIN_Z_M
from nero_agent.mujoco_preflight import CollisionScene, validate, retime_interpolation, bezier_segment
from nero_agent.ros_backend import RosBackend
from nero_agent.stream_guard import StreamGuard
from tests.test_mujoco_preflight import ROOT, config, description, point
from tests.test_experiment import fake_hardware, DEMO_START


def settings():
    value = json.loads((ROOT / 'examples/nero-agent.mock.json').read_text())
    value['motion_profile'] = 'mujoco_floor'
    return Settings.parse(value)


def trajectory(start, goal, duration=10):
    return NS(joint_names=list(JOINTS), points=[point(start, 0), point(goal, duration)])


class FloorMotionTests(unittest.TestCase):
    def test_profile_is_continuous_and_keeps_preflight(self):
        s = settings()
        self.assertTrue(s.floor_guard and s.mujoco_preflight)
        self.assertFalse(s.segmented_execution)
        self.assertEqual((s.max_velocity, s.max_acceleration), (.2, .5))
        self.assertEqual(s.max_excursion, FLOOR_EXCURSION_RAD)
        self.assertFalse(config().floor_guard)

    def test_real_interior_move_accelerates_without_changing_path(self):
        q = [0.] * 7
        goal = [.3] + [0.] * 6  # Previously at least four stop-and-settle legs.
        t = trajectory(q, goal, 40)
        original = bezier_segment(*t.points, 40)
        result = retime_interpolation(t, settings(), allow_speedup=True)
        self.assertLess(result['interpolation_time_scale'], .1)
        end = t.points[-1].time_from_start
        duration = end.sec + end.nanosec*1e-9
        np.testing.assert_allclose(bezier_segment(*t.points, duration), original, atol=1e-9)
        backend = RosBackend.__new__(RosBackend)
        backend.settings, backend.initial = settings(), q
        result = backend._mujoco_preflight(description(), NS(joint_trajectory=t),
                                           {'gripper_width_m': .04})['mujoco_preflight']
        self.assertEqual(result['speed_region'], 'interior')
        self.assertEqual(result['execution_mode'], 'continuous')
        self.assertEqual(len(t.points), 2)
        self.assertLess(duration, 3.)
        self.assertGreaterEqual(result['minimum_floor_clearance_bound_m'], .04)

    def test_clear_floor_move_is_interior(self):
        q = [.6, .8, -.034, .729, .037, -.007, -.222]
        goal = q.copy()
        goal[0] += .002
        t = trajectory(q, goal)
        retime_interpolation(t, settings(), allow_speedup=True)
        backend = RosBackend.__new__(RosBackend)
        backend.settings, backend.initial = settings(), q
        result = backend._mujoco_preflight(description(), NS(joint_trajectory=t),
                                           {'gripper_width_m': .04})['mujoco_preflight']
        self.assertEqual(result['speed_region'], 'interior')
        self.assertEqual(result['velocity_limit_rad_s'], .2)
        self.assertEqual(result['acceleration_limit_rad_s2'], .5)
        validate(description(), t, {'gripper_width_m': .04}, q,
                 settings())
        self.assertEqual(len(t.points), 2)

    def test_floor_in_base_frame_includes_fingers_and_all_moving_links(self):
        s = CollisionScene(description(), config().collision_boxes, .04)
        q = [0.] * 7
        s.distances(q)
        checked = {s.names[i] for i in s.floor_geoms}
        self.assertNotIn('preflight_base_link_0', checked)
        for name in ['link%d' % i for i in range(1, 8)] + ['gripper_link1', 'gripper_link2', 'gripper_base']:
            self.assertIn('preflight_' + name + '_0', checked)
        original = s.floor_heights()
        root = ET.fromstring(description())
        joint = root.find('./joint[@name="world_to_base_link"]')
        origin = joint.find('origin')
        if origin is None:
            origin = ET.SubElement(joint, 'origin')
        origin.set('xyz', '1 2 3')
        origin.set('rpy', '.1 -.2 .4')
        shifted = CollisionScene(ET.tostring(root, encoding='unicode'), config().collision_boxes, .04)
        shifted.distances(q)
        np.testing.assert_allclose(shifted.floor_heights(), original, atol=1e-6)

    def test_floor_crossing_between_safe_endpoints_is_rejected(self):
        current = [0.]
        def distances(q):
            current[0] = q[0]
            return np.array([1.])
        scene = NS(ranges=np.array([[-7., 7.]] * 7), motion=np.zeros((1, 7)),
                   distances=distances, first=[0], second=[1], names=['arm', 'tool'],
                   floor_motion=np.array([[1., 0, 0, 0, 0, 0, 0]]),
                   floor_heights=lambda: np.array([(current[0] - .5)**2 - .05]))
        self.assertGreater(.5**2 - .05, 0.)  # Both endpoint heights are positive.
        with patch('nero_agent.mujoco_preflight.CollisionScene', return_value=scene):
            with self.assertRaisesRegex(AgentError, 'crosses floor minimum z -0.040000'):
                validate('model', trajectory([0.]*7, [1.]+[0.]*6),
                         {'gripper_width_m': .04}, [0.]*7, settings())

    def test_floor_tracking_allowance_is_required(self):
        scene = NS(ranges=np.array([[-7., 7.]] * 7), motion=np.zeros((1, 7)),
                   distances=lambda q: np.array([1.]), first=[0], second=[1], names=['arm', 'tool'],
                   floor_motion=np.array([[1., 0, 0, 0, 0, 0, 0]]),
                   floor_heights=lambda: np.array([-.03]))
        with patch('nero_agent.mujoco_preflight.CollisionScene', return_value=scene):
            with self.assertRaisesRegex(AgentError, 'floor clearance bound -0.045'):
                validate('model', trajectory([0.]*7, [0.]*7),
                         {'gripper_width_m': .04}, [0.]*7, settings())

    def test_self_collision_still_blocks_high_above_floor(self):
        scene = NS(ranges=np.array([[-7., 7.]] * 7), motion=np.zeros((1, 7)),
                   distances=lambda q: np.array([-.01]), first=[0], second=[1], names=['arm', 'tool'],
                   floor_motion=np.zeros((1, 7)), floor_heights=lambda: np.array([1.]))
        with patch('nero_agent.mujoco_preflight.CollisionScene', return_value=scene):
            with self.assertRaisesRegex(AgentError, 'clearance rejected'):
                validate('model', trajectory([0.]*7, [0.]*7),
                         {'gripper_width_m': .04}, [0.]*7, settings())

    def test_acceleration_raise_is_explicit_and_verified_without_motion(self):
        hardware = fake_hardware(DEMO_START)
        hardware.robot.accelerations = [.15]*7
        self.assertEqual(hardware.configure_acceleration(.5), [.15]*7)
        self.assertEqual(hardware.configure_acceleration(.5, allow_increase=True), [.5]*7)
        self.assertEqual(len(hardware.robot.acceleration_writes), 7)
        self.assertFalse(hardware.robot.commands)

    def test_interior_stream_allows_speed_but_preserves_tracking_and_watchdog(self):
        guard = StreamGuard([0.]*7, .04, 0, .30, FLOOR_EXCURSION_RAD,
                            command_velocity_limit=.25, position_velocity_limit=.50)
        guard.command([0.]*7, [0.]*7, 100., 100., 0.)
        q = [.004] + [0.]*6
        guard.command(q, q, 100.02, 100.02, .02)
        guard.check_feedback(q, [.2]+[0.]*6, .04, .02)
        guard.check_feedback(q, [.2]+[0.]*6, .04, .03)
        with self.assertRaisesRegex(AgentError, 'tracking'):
            guard.command([.031]+[0.]*6, [0.]*7, 100.04, 100.04, .04)
        with self.assertRaisesRegex(AgentError, 'timed out'):
            guard.check_feedback(q, [0.]*7, .04, .6)
        with self.assertRaisesRegex(AgentError, 'velocity'):
            guard.command([.01]+[0.]*6, [.01]+[0.]*6, 100.03, 100.03, .03)

    def test_driver_consumes_region_once_and_sets_matching_limits(self):
        from contextlib import contextmanager
        from unittest.mock import MagicMock
        from nero_agent import driver
        state = NS(joints_rad=[0.]*7, velocities_rad_s=[0.]*7)
        hardware = NS(robot=MagicMock(), stationary=MagicMock(return_value=state),
                      configure_acceleration=MagicMock())
        @contextmanager
        def connect():
            yield hardware
        class Node:
            def __init__(self, *args, **kwargs): pass
            def create_client(self, *args): return MagicMock()
            def create_publisher(self, *args): return MagicMock()
            def create_subscription(self, *args): pass
            def create_service(self, *args): pass
            def create_timer(self, *args): pass
            def get_logger(self): return MagicMock()
            def destroy_node(self): pass
        def spin(node):
            node.tick = lambda: None
            node.gripper = lambda: (.04, 100.)
            # A floor driver must never silently default to the fast mode.
            self.assertFalse(node.gate(NS(data=True), NS()).success)
            hardware.configure_acceleration.assert_not_called()
            for near, acceleration, command_limit, measured_limit in (
                    (False, .5, .25, .30), (True, .03, .10, .10)):
                self.assertTrue(node.set_floor_region(NS(data=near), NS()).success)
                self.assertTrue(node.gate(NS(data=True), NS()).success)
                self.assertIsNone(node.floor_near)
                if near:
                    hardware.configure_acceleration.assert_called_with(acceleration)
                else:
                    hardware.configure_acceleration.assert_called_with(acceleration, allow_increase=True)
                self.assertEqual(node.guard.position_velocity_limit, .15 if near else .50)
                self.assertEqual(node.guard.command_velocity_limit, command_limit)
                self.assertEqual(node.guard.motor_velocity_limit, measured_limit)
                self.assertFalse(node.set_floor_region(NS(data=not near), NS()).success)
                self.assertTrue(node.gate(NS(data=False), NS()).success)
                self.assertFalse(node.gate(NS(data=True), NS()).success)
            hardware.robot.move_j.assert_not_called()
        modules = {'rclpy': NS(init=lambda: None, spin=spin, ok=lambda: True, shutdown=lambda: None),
                   'rclpy.node': NS(Node=Node), 'sensor_msgs.msg': NS(JointState=NS),
                   'control_msgs.msg': NS(JointTrajectoryControllerState=NS),
                   'std_msgs.msg': NS(Empty=NS, String=NS),
                   'std_srvs.srv': NS(SetBool=NS, Trigger=NS),
                   'action_msgs.srv': NS(CancelGoal=NS(Request=NS)),
                   'nero_experiment.hardware': NS(connect=connect, MAX_JOINT_SNAPSHOT_AGE_S=.080)}
        with patch.dict('sys.modules', modules), patch('sys.argv', ['driver', '--floor-motion']), \
                patch('nero_agent.abort_policy.require_verified_controlled_abort'):
            driver.main()


    def test_negative_floor_boundary_and_shifted_slow_band(self):
        for height, allowed, region in ((-.040001, False, None), (-.04, True, 'near_floor'),
                                        (-.02, True, 'near_floor'), (.01, True, 'interior')):
            with self.subTest(height=height):
                scene = NS(ranges=np.array([[-7., 7.]] * 7), motion=np.zeros((1, 7)),
                           distances=lambda q: np.array([1.]), first=[0], second=[1],
                           names=['arm', 'tool'], inertial_placeholders=[],
                           floor_motion=np.zeros((1, 7)), floor_heights=lambda: np.array([height]))
                q = [0.]*7
                t = trajectory(q, [.1]+[0.]*6)
                retime_interpolation(t, settings(), allow_speedup=True)
                backend = RosBackend.__new__(RosBackend)
                backend.settings, backend.initial = settings(), q
                with patch('nero_agent.mujoco_preflight.CollisionScene', return_value=scene):
                    if not allowed:
                        with self.assertRaisesRegex(AgentError, 'floor minimum z'):
                            backend._mujoco_preflight('model', NS(joint_trajectory=t), {'gripper_width_m': .04})
                    else:
                        result = backend._mujoco_preflight('model', NS(joint_trajectory=t),
                                                          {'gripper_width_m': .04})['mujoco_preflight']
                        self.assertEqual(result['speed_region'], region)
                        self.assertEqual(result['floor_minimum_z_m'], FLOOR_MIN_Z_M)
                        self.assertEqual(len(t.points), 2)
                        if region == 'near_floor':
                            validate('model', t, {'gripper_width_m': .04}, q,
                                     replace(settings(), max_velocity=.02, max_acceleration=.03))


    def test_position_velocity_accepts_reported_spike_but_rejects_over_half_rad_s(self):
        for speed, allowed in ((-.327269, True), (.49, True), (-.501, False)):
            with self.subTest(speed=speed):
                guard = StreamGuard([0.]*7, .04, 0, .30, FLOOR_EXCURSION_RAD,
                                    command_velocity_limit=.25, position_velocity_limit=.50)
                guard.check_feedback([0.]*7, [0.]*7, .04, 0.,
                                     position_timestamps=[100.]*4)
                q = [0.]*6 + [speed*.02]
                if allowed:
                    guard.check_feedback(q, [0.]*7, .04, .02,
                                         position_timestamps=[100.02]*4)
                else:
                    with self.assertRaisesRegex(AgentError, 'Position-derived velocity exceeded 0.500'):
                        guard.check_feedback(q, [0.]*7, .04, .02,
                                             position_timestamps=[100.02]*4)
