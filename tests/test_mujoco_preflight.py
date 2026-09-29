"""Offline collision and interpolation checks; no ROS or hardware."""
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import numpy as np

from nero_agent.core import AgentError, Settings, JOINTS
from nero_agent.mujoco_preflight import CollisionScene, bezier_segment, split, validate, retime_interpolation
from nero_agent.ros_backend import RosBackend
from nero_agent.stream_guard import StreamGuard

ROOT = Path(__file__).resolve().parents[1]


def description():
    path = ROOT / 'models/nero/nero_description.urdf'
    root = ET.parse(path).getroot()
    for mesh in root.findall('.//mesh'):
        mesh.set('filename', str(path.parent / mesh.get('filename')))
    return ET.tostring(root, encoding='unicode')


def config():
    value = json.loads((ROOT / 'examples/nero-agent.mock.json').read_text())
    value['motion_profile'] = 'mujoco_large'
    value['collision_boxes'] = [{'id': 'table', 'size_m': [2, 2, .04], 'center_m': [0, 0, -.03]}]
    return Settings.parse(value)


def point(q, t):
    return NS(positions=list(q), velocities=[0.]*7, accelerations=[0.]*7,
              time_from_start=NS(sec=t, nanosec=0))


class PreflightTests(unittest.TestCase):
    def test_retiming_bounds_between_waypoint_acceleration_without_changing_curve(self):
        from nero_agent.trajectory import validate_trajectory
        goal = list(self.q)
        goal[0] += .001
        first, last = point(self.q, 0), point(goal, 0)
        last.time_from_start.nanosec = 100000000
        trajectory = NS(joint_names=list(JOINTS), points=[first, last])
        # Zero waypoint derivatives pass the old check despite interior overshoot.
        validate_trajectory(trajectory, self.q, goal, self.q, self.settings)
        original = bezier_segment(first, last, .1)
        timing = retime_interpolation(trajectory, self.settings)
        self.assertGreater(timing['interpolation_time_scale'], 1.)
        self.assertGreater(timing['original_interpolation_acceleration_bound_rad_s2'], .15)
        dt = last.time_from_start.sec + last.time_from_start.nanosec*1e-9
        np.testing.assert_allclose(bezier_segment(first, last, dt), original, atol=1e-10)
        validate_trajectory(trajectory, self.q, goal, self.q, self.settings)
        self.assertEqual(validate(description(), trajectory, self.before, self.q,
                                  self.settings)['status'], 'passed')
        self.assertEqual(retime_interpolation(trajectory, self.settings)['interpolation_time_scale'], 1.)

    def test_retiming_rejects_timeout(self):
        from dataclasses import replace
        first, last = point(self.q, 0), point([q + .001 for q in self.q], 0)
        last.time_from_start.nanosec = 100000000
        trajectory = NS(points=[first, last])
        with self.assertRaisesRegex(AgentError, 'retiming exceeds execution timeout'):
            retime_interpolation(trajectory, replace(self.settings, timeout=.11))
        self.assertEqual(last.time_from_start.nanosec, 100000000)

    def test_deep_subdivision_preserves_acceleration_and_rejects_real_overrun(self):
        from unittest.mock import MagicMock
        for acceleration in (.135, .16):
            with self.subTest(acceleration=acceleration):
                scene = NS(ranges=np.array([[-3., 3.]]*7), motion=np.full((1, 7), .001),
                           first=[0], second=[1], names=['a', 'b'], inertial_placeholders=[])
                calls = [0]
                def distances(q):
                    calls[0] += 1
                    # Force sixteen subdivisions before clearance succeeds.
                    return np.array([.0031 if calls[0] <= 16 else .1])
                scene.distances = MagicMock(side_effect=distances)
                dt = .1
                first = point([2.]*7, 0)
                last = point([2. + .5*acceleration*dt*dt]*7, 0)
                last.time_from_start.nanosec = 100000000
                first.accelerations = last.accelerations = [acceleration]*7
                last.velocities = [acceleration*dt]*7
                trajectory = NS(joint_names=list(JOINTS), points=[first, last])
                with patch('nero_agent.mujoco_preflight.CollisionScene', return_value=scene):
                    if acceleration < .15:
                        result = validate('test', trajectory, {'gripper_width_m': .04}, [2.]*7, config())
                        self.assertEqual(result['status'], 'passed')
                        self.assertGreater(result['intervals_checked'], 16)
                    else:
                        with self.assertRaisesRegex(AgentError, 'acceleration bound'):
                            validate('test', trajectory, {'gripper_width_m': .04}, [2.]*7, config())

    def setUp(self):
        self.q = [.6, 1.236, -.034, .729, .037, -.007, -.222]
        self.before = {'joints_rad': self.q, 'gripper_width_m': .099631}
        self.settings = config()
        goal = self.q.copy()
        goal[0] += .002
        self.trajectory = NS(joint_names=list(JOINTS), points=[point(self.q, 0), point(goal, 2)])

    def test_profile_is_explicit_and_bounded(self):
        self.assertEqual(self.settings.max_excursion, 1.2)
        self.assertEqual(self.settings.timeout, 90.)
        self.assertTrue(self.settings.mujoco_preflight)
        self.assertEqual(Settings('mock', {}, ()).max_excursion, .3)
        c = json.loads((ROOT / 'examples/nero-agent.mock.json').read_text())
        c['motion_profile'] = 'unlimited'
        with self.assertRaises(AgentError):
            Settings.parse(c)

    def test_quintic_control_points_and_subdivision(self):
        first, last = point([0.]*7, 0), point([1.]*7, 2)
        b = bezier_segment(first, last, 2)
        np.testing.assert_allclose(b[:, 0], [0, 0, 0, 1, 1, 1])
        left, right = split(b)
        np.testing.assert_allclose(left[-1], .5)
        np.testing.assert_allclose(left[-1], right[0])
        np.testing.assert_allclose(5*(b[1]-b[0])/2, first.velocities)

    def test_real_model_small_path_passes_with_measured_gripper(self):
        result = validate(description(), self.trajectory, self.before, self.q, self.settings)
        self.assertEqual(result['status'], 'passed')
        self.assertGreaterEqual(result['minimum_clearance_bound_m'], .003)
        scene = CollisionScene(description(), self.settings.collision_boxes, .04)
        index = scene.model.joint('gripper_joint1').id
        self.assertAlmostEqual(scene.data.qpos[scene.model.jnt_qposadr[index]], .02)

    def test_obstacle_on_arm_blocks(self):
        scene = CollisionScene(description(), self.settings.collision_boxes, .099631)
        scene.distances(self.q)
        geom = scene.names.index('preflight_link4_0')
        box = {'id': 'obstruction', 'size_m': [.2]*3,
               'center_m': scene.data.geom_xpos[geom].tolist()}
        from dataclasses import replace
        settings = replace(self.settings, collision_boxes=(*self.settings.collision_boxes, box))
        with self.assertRaisesRegex(AgentError, 'clearance rejected'):
            validate(description(), self.trajectory, self.before, self.q, settings)

    def test_missing_mesh_blocks(self):
        xml = description().replace('base_link.stl', 'missing_mesh.stl')
        with self.assertRaisesRegex(AgentError, 'mesh must resolve'):
            validate(xml, self.trajectory, self.before, self.q, self.settings)

    def test_vendor_virtual_gripper_without_inertia_loads_without_geometry_change(self):
        root = ET.fromstring(description())
        link = root.find('./link[@name="gripper_link"]')
        link.remove(link.find('inertial'))
        xml = ET.tostring(root, encoding='unicode')
        original = CollisionScene(description(), self.settings.collision_boxes, .04)
        imported = CollisionScene(xml, self.settings.collision_boxes, .04)
        self.assertEqual(imported.inertial_placeholders, ['gripper_link'])
        self.assertEqual(imported.names, original.names)
        np.testing.assert_allclose(imported.ranges, original.ranges, atol=0, rtol=0)
        np.testing.assert_allclose(imported.distances(self.q), original.distances(self.q), atol=1e-12)
        result = validate(xml, self.trajectory, self.before, self.q, self.settings)
        self.assertEqual(result['kinematic_inertial_placeholders'], ['gripper_link'])
        self.assertIsNone(ET.fromstring(xml).find('./link[@name="gripper_link"]/inertial'))

    def test_placeholder_is_not_applied_to_an_unrecognized_gripper(self):
        root = ET.fromstring(description())
        link = root.find('./link[@name="gripper_link"]')
        link.remove(link.find('inertial'))
        root.find('./joint[@name="gripper"]').set('type', 'revolute')
        with self.assertRaisesRegex(AgentError, 'Unrecognized massless gripper'):
            CollisionScene(ET.tostring(root, encoding='unicode'), self.settings.collision_boxes, .04)

    def test_fixed_base_transform_preserves_robot_obstacle_clearances(self):
        original = CollisionScene(description(), self.settings.collision_boxes, .04)
        root = ET.fromstring(description())
        joint = root.find('./joint[@name="world_to_base_link"]')
        origin = joint.find('origin')
        if origin is None:
            origin = ET.SubElement(joint, 'origin')
        origin.set('xyz', '1 2 3')
        origin.set('rpy', '0.1 -0.2 0.4')
        shifted = CollisionScene(ET.tostring(root, encoding='unicode'),
                                 self.settings.collision_boxes, .04)
        np.testing.assert_allclose(original.distances(self.q), shifted.distances(self.q),
                                   atol=1e-6, rtol=0)

    def test_interpolation_velocity_not_just_waypoints_is_checked(self):
        self.trajectory.points[1].positions[0] += .3
        self.trajectory.points[1].time_from_start.sec = 1
        with self.assertRaisesRegex(AgentError, 'could not certify|budget') as raised:
            validate(description(), self.trajectory, self.before, self.q, self.settings)
        self.assertRegex(str(raised.exception), r'joint1 (velocity|acceleration) bound')
        self.assertIn('time [', str(raised.exception))

    def test_clearance_diagnostic_identifies_tracking_limited_pair(self):
        scene = NS(ranges=np.array([[-10., 10.]]*7),
                   distances=lambda q: np.array([.004, .010]),
                   motion=np.array([[0.]*7, [1.]*7]),
                   names=['near_a', 'near_b', 'tracking_a', 'tracking_b'],
                   first=np.array([0, 2]), second=np.array([1, 3]))
        with patch('nero_agent.mujoco_preflight.CollisionScene', return_value=scene):
            with self.assertRaisesRegex(AgentError, 'could not certify') as raised:
                validate('model', self.trajectory, self.before, self.q, self.settings)
        message = str(raised.exception)
        for detail in ('pair tracking_a / tracking_b', 'nominal clearance 0.010000000 m',
                       'tracking deduction 0.105000000 m', 'clearance lower bound',
                       'motion-bound violations: none', 'waypoint interval 0', 'depth 16',
                       '2.000000000] s'):
            self.assertIn(detail, message)

    def test_backend_requires_preflight_and_propagates_failure(self):
        backend = RosBackend.__new__(RosBackend)
        backend.settings, backend.initial = self.settings, self.q
        with patch('nero_agent.mujoco_preflight.validate', side_effect=AgentError('blocked')):
            with self.assertRaisesRegex(AgentError, 'blocked'):
                backend._mujoco_preflight('model', NS(joint_trajectory=self.trajectory), self.before)

    def test_large_driver_limit_still_rejects_overrun(self):
        guard = StreamGuard([0.]*7, .04, 0, max_excursion=1.2)
        guard.check_feedback([1.1]+[0.]*6, [0.]*7, .04, .01)
        with self.assertRaisesRegex(AgentError, 'excursion'):
            guard.check_feedback([1.211]+[0.]*6, [0.]*7, .04, .02)
        guard.check_feedback([1.1]+[0.]*6, [.2]+[0.]*6, .04, .03)
        with self.assertRaisesRegex(AgentError, 'velocity'):
            guard.check_feedback([1.1]+[0.]*6, [.2]+[0.]*6, .04, .04)

    def test_segmented_route_passes_real_model_preflight(self):
        from dataclasses import replace
        backend = RosBackend.__new__(RosBackend)
        backend.settings = replace(self.settings, segmented_execution=True,
                                   max_excursion=3.14, timeout=600.)
        backend.initial = self.q
        backend.preflight_description = description()
        route, summary = backend._prepare_segments(NS(joint_trajectory=self.trajectory), self.before)
        self.assertEqual(len(route.segments), 1)
        self.assertEqual(summary['segments'][0]['mujoco_preflight']['status'], 'passed')


if __name__ == '__main__':
    unittest.main()
