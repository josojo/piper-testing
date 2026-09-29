"""Offline tests of segmented planning and fail-closed sequencing."""
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock

import numpy as np

from nero_agent.core import AgentError, JOINTS, Settings, distance
from nero_agent.mujoco_preflight import bezier_segment
from nero_agent.ros_backend import RosBackend
from nero_agent.segmented import (SegmentedRoute, make_segments, retime_leg,
                                  require_stopped_at, require_supported_scene, STOPPED_SPEED)


def trajectory(values):
    return NS(joint_trajectory=NS(joint_names=list(JOINTS), points=[
        NS(positions=[x]+[0.]*6, velocities=[0.]*7, accelerations=[0.]*7,
           time_from_start=NS(sec=i, nanosec=0)) for i, x in enumerate(values)]))


def state(q):
    return {'joints_rad': list(q), 'velocities_rad_s': [0.]*7, 'gripper_width_m': .04}


class SegmentTests(unittest.TestCase):
    def test_sparse_and_dense_routes_have_bounded_continuous_legs(self):
        for values in ([0., 1.65], list(np.linspace(0, 1.65, 100)), [0., .2, -.2, .1]):
            segments = make_segments(trajectory(values))
            previous = [values[0]]+[0.]*6
            for leg in segments:
                first, last = leg.joint_trajectory.points
                self.assertEqual(first.positions, previous)
                self.assertLessEqual(distance(first.positions, last.positions), .090000001)
                dt = last.time_from_start.sec + last.time_from_start.nanosec*1e-9
                self.assertLessEqual(dt, 15)
                b = bezier_segment(first, last, dt)
                # Sample actual quintic derivative, independently of coefficient bounds.
                from math import comb
                v = 5*np.diff(b, axis=0)/dt
                a = 4*np.diff(v, axis=0)/dt
                for t in np.linspace(0, 1, 101):
                    velocity = sum(comb(4,i)*t**i*(1-t)**(4-i)*v[i] for i in range(5))
                    acceleration = sum(comb(3,i)*t**i*(1-t)**(3-i)*a[i] for i in range(4))
                    self.assertLessEqual(max(abs(velocity)), .020000001)
                    self.assertLessEqual(max(abs(acceleration)), .030000001)
                self.assertEqual(first.velocities, [0.]*7)
                self.assertEqual(last.velocities, [0.]*7)
                previous = last.positions
            self.assertEqual(previous, [values[-1]]+[0.]*6)

    def test_alignment_preserves_endpoint_and_retimes(self):
        leg = make_segments(trajectory([0., .08]))[0]
        endpoint = list(leg.joint_trajectory.points[-1].positions)
        leg = retime_leg(leg, [-.004]+[0.]*6, endpoint)
        self.assertEqual(leg.joint_trajectory.points[-1].positions, endpoint)
        with self.assertRaisesRegex(AgentError, 'per-step'):
            retime_leg(leg, [-.03]+[0.]*6, endpoint)

    def test_boundary_requires_arrival_gripper_and_standstill(self):
        for kind in ('position', 'gripper', 'velocity'):
            actual = state([0.]*7)
            if kind == 'position': actual['joints_rad'][0] = .006
            if kind == 'gripper': actual['gripper_width_m'] += .002
            if kind == 'velocity': actual['velocities_rad_s'][0] = STOPPED_SPEED+.001
            with self.subTest(kind=kind), self.assertRaises(AgentError):
                require_stopped_at(actual, [0.]*7, .04)

    def test_extra_or_changed_scene_geometry_blocks_shortcuts(self):
        box = {'id': 'table', 'size_m': [2., 2., .04], 'center_m': [0., 0., -.03]}
        obj = NS(id='nero_agent_table', header=NS(frame_id='base_link'), meshes=[], planes=[],
                 primitives=[NS(type=1, dimensions=box['size_m'])],
                 primitive_poses=[NS(position=NS(x=0., y=0., z=-.03),
                                     orientation=NS(x=0., y=0., z=0., w=1.))])
        scene = NS(robot_state=NS(attached_collision_objects=[]),
                   world=NS(collision_objects=[obj], octomap=NS(octomap=NS(data=[]))))
        require_supported_scene(scene, [box])
        for case in ('attached', 'octomap', 'unknown', 'moved'):
            changed = deepcopy(scene)
            if case == 'attached': changed.robot_state.attached_collision_objects.append(object())
            if case == 'octomap': changed.world.octomap.octomap.data.append(1)
            if case == 'unknown': changed.world.collision_objects.append(deepcopy(obj))
            if case == 'moved': changed.world.collision_objects[0].primitive_poses[0].position.z += .1
            with self.subTest(case=case), self.assertRaises(AgentError):
                require_supported_scene(changed, [box])

    def backend(self):
        b = RosBackend.__new__(RosBackend)
        b.settings = replace(Settings('mock', {}, ()), segmented_execution=True,
                             mujoco_preflight=True, max_excursion=3.14, timeout=600.)
        b.initial = [0.]*7
        b.plans = {}
        b.stop = MagicMock()
        b._scene_digest = lambda: 'scene'
        b._check_segment = MagicMock(return_value={'duration_s': 2.})
        b.preflight_description = 'loaded model'
        return b

    def test_moveit_world_frame_and_split_object_pose(self):
        def pose(x=0., y=0., z=0., q=(0., 0., 0., 1.)):
            return NS(position=NS(x=x, y=y, z=z),
                      orientation=NS(x=q[0], y=q[1], z=q[2], w=q[3]))
        box = {'id': 'table', 'size_m': [2., 2., .04], 'center_m': [0., 0., -.03]}
        xml = ('<robot name="test"><link name="world"/><link name="base_link"/>'
               '<joint name="mount" type="fixed"><parent link="world"/>'
               '<child link="base_link"/><origin xyz="1 2 3" rpy="0 0 1.5707963267948966"/>'
               '</joint></robot>')
        # world_T_object includes mount rotation and an object-local offset.
        obj = NS(id='nero_agent_table', header=NS(frame_id='world'), meshes=[], planes=[],
                 pose=pose(1., 2., 3., (0., 0., 2**-.5, 2**-.5)),
                 primitives=[NS(type=1, dimensions=box['size_m'])],
                 primitive_poses=[pose(z=-.03)])
        scene = NS(robot_state=NS(attached_collision_objects=[]),
                   world=NS(collision_objects=[obj], octomap=NS(octomap=NS(data=[]))))
        require_supported_scene(scene, [box], xml)
        obj.header.frame_id = '/world'
        require_supported_scene(scene, [box], xml)
        # A genuine displacement must still be rejected after transformation.
        obj.pose.position.x += .01
        with self.assertRaisesRegex(AgentError, 'after transforming'):
            require_supported_scene(scene, [box], xml)
        obj.pose.position.x -= .01
        for invalid in ('missing_frame', 'moving_frame', 'zero_quaternion', 'mesh', 'size'):
            changed = deepcopy(scene)
            item = changed.world.collision_objects[0]
            model = xml
            if invalid == 'missing_frame': item.header.frame_id = 'unknown'
            if invalid == 'moving_frame': model = xml.replace('type="fixed"', 'type="revolute"')
            if invalid == 'zero_quaternion': item.pose.orientation = NS(x=0., y=0., z=0., w=0.)
            if invalid == 'mesh': item.meshes = [object()]
            if invalid == 'size': item.primitives[0].dimensions = [3., 2., .04]
            with self.subTest(invalid=invalid), self.assertRaises(AgentError):
                require_supported_scene(changed, [box], model)

    def test_rotated_object_and_primitive_compose_in_base_frame(self):
        box = {'id': 'table', 'size_m': [2., 2., .04], 'center_m': [0., 0., -.03]}
        obj = NS(id='nero_agent_table', header=NS(frame_id='base_link'), meshes=[], planes=[],
                 pose=NS(position=NS(x=1., y=0., z=0.),
                         orientation=NS(x=0., y=0., z=2**-.5, w=2**-.5)),
                 primitives=[NS(type=1, dimensions=box['size_m'])],
                 primitive_poses=[NS(position=NS(x=0., y=1., z=-.03),
                                     orientation=NS(x=0., y=0., z=-2**-.5, w=2**-.5))])
        scene = NS(robot_state=NS(attached_collision_objects=[]),
                   world=NS(collision_objects=[obj], octomap=NS(octomap=NS(data=[]))))
        require_supported_scene(scene, [box])

    def test_all_legs_checked_before_plan_is_cached(self):
        b = self.backend()
        raw = trajectory([0., .25])
        b.state = lambda: state([0.]*7)
        b._plan_from_state = lambda goal, before: (raw, 'scene', {})
        b._check_segment.side_effect = [{ 'duration_s': 2.}, AgentError('collision')]
        with self.assertRaisesRegex(AgentError, 'collision'):
            b.plan([.25]+[0.]*6)
        self.assertEqual(b.plans, {})

    def test_verified_segments_advance_and_report(self):
        b = self.backend()
        route = SegmentedRoute(make_segments(trajectory([0., .2])), 'loaded model')
        actual = [state([0.]*7)]
        b.state = lambda: deepcopy(actual[0])
        b.segment_callback = MagicMock()
        def execute(plan):
            leg, before, endpoint, digest = b.plans.pop(plan['token'])
            self.assertTrue(b.executing_segment)
            actual[0] = state(endpoint)
            return deepcopy(actual[0])
        b.execute = MagicMock(side_effect=execute)
        result = b._execute_segments(route, actual[0], [.2]+[0.]*6, 'scene')
        self.assertEqual(result['joints_rad'][0], .2)
        self.assertEqual(b.execute.call_count, len(route.segments))
        self.assertEqual(b.segment_callback.call_count, len(route.segments))
        b.stop.assert_not_called()

    def test_divergence_scene_change_or_failed_preflight_prevents_next_action(self):
        for failure in ('divergence', 'scene', 'preflight', 'action'):
            b = self.backend()
            route = SegmentedRoute(make_segments(trajectory([0., .2])), 'model')
            initial = state([0.]*7)
            actual = [initial]
            b.state = lambda: deepcopy(actual[0])
            def execute(plan):
                leg, before, endpoint, digest = b.plans.pop(plan['token'])
                if failure == 'action':
                    raise AgentError('action rejected')
                actual[0] = state(endpoint)
                if failure == 'divergence': actual[0]['joints_rad'][0] += .02
                if failure == 'scene': b._scene_digest = lambda: 'changed'
                if failure == 'preflight': b._check_segment.side_effect = AgentError('collision')
                return deepcopy(actual[0])
            b.execute = MagicMock(side_effect=execute)
            with self.subTest(failure=failure), self.assertRaises(AgentError):
                b._execute_segments(route, initial, [.2]+[0.]*6, 'scene')
            self.assertEqual(b.execute.call_count, 1)
            self.assertFalse(b.executing_segment)
            b.stop.assert_called_once()


if __name__ == '__main__':
    unittest.main()
