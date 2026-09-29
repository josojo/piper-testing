import math
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

from nero_agent.core import AgentError, Settings
from nero_agent.ros_backend import RosBackend, configure_goal_orientation
from tests.test_agent import config


class PoseOptionsTests(unittest.TestCase):
    def test_down_free_spin_and_fixed_defaults(self):
        for mode, yaw in [('fixed', .15), ('camera_down_free_yaw', math.pi)]:
            data = config()
            pose = {'position_m': [0., 0., .2], 'orientation_xyzw': [1., 0., 0., 0.],
                    'orientation_mode': mode}
            data['named_poses'] = {'photo': pose}
            settings = Settings.parse(data)
            self.assertEqual(settings.resolve([0.]*7)['photo'], pose)
            constraint = NS(XYZ_EULER_ANGLES=0)
            configure_goal_orientation(constraint, pose)
            self.assertEqual(constraint.parameterization, 0)
            self.assertEqual(constraint.absolute_x_axis_tolerance, .08)
            self.assertEqual(constraint.absolute_y_axis_tolerance, .08)
            self.assertEqual(constraint.absolute_z_axis_tolerance, yaw)
        constraint = NS(XYZ_EULER_ANGLES=0)
        configure_goal_orientation(constraint, {})
        self.assertEqual(constraint.absolute_z_axis_tolerance, .15)

    def test_rejects_wrong_direction_and_unknown_mode(self):
        for mode, quat in [('camera_down_free_yaw', [0., 0., 0., 1.]),
                           ('free_everything', [1., 0., 0., 0.])]:
            data = config()
            data['named_poses'] = {'photo': {'position_m': [0., 0., .2],
                'orientation_xyzw': quat, 'orientation_mode': mode}}
            with self.assertRaises(AgentError):
                Settings.parse(data)

    def test_capture_fk_uses_exact_measured_snapshot(self):
        backend = RosBackend.__new__(RosBackend)
        backend.settings = NS(namespace='/nero')
        backend.node = MagicMock()
        request = NS(header=NS(), robot_state=NS(joint_state=NS()))
        service = NS(Request=lambda: request)
        pose = NS(position=NS(x=.1, y=.2, z=.3), orientation=NS(x=1., y=0., z=0., w=0.))
        backend._call = MagicMock(return_value=NS(error_code=NS(val=1), fk_link_names=['tcp_link'],
            pose_stamped=[NS(header=NS(frame_id='base_link'), pose=pose)]))
        state = {'joints_rad': [.1]*7, 'gripper_width_m': .04, 'captured_at_unix': 123.}
        with patch.dict('sys.modules', {'moveit_msgs.srv': NS(GetPositionFK=service)}):
            result = backend.tcp_pose(state)
            self.assertEqual(request.robot_state.joint_state.position, [.1]*7+[.04])
            self.assertEqual(result['position_m'], [.1, .2, .3])
            self.assertEqual(result['captured_at_unix'], 123.)
            self.assertEqual(request.header.frame_id, 'base_link')
            backend._call.return_value.error_code.val = -1
            with self.assertRaises(AgentError):
                backend.tcp_pose(state)


if __name__ == '__main__':
    unittest.main()
