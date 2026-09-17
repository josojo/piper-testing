"""Offline tests: no ROS graph, LLM requests, CAN sockets or motor commands."""
from contextlib import redirect_stdout, redirect_stderr
from array import array
from copy import deepcopy
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
import tempfile
import time
import unittest
from unittest.mock import patch, MagicMock

from nero_agent.core import (AgentError, DirectPoseChooser, JOINTS, OfflineBackend, ScriptedChooser,
                             Settings, run_loop, strict_json, validate_action)
from nero_agent.__main__ import main
from nero_agent.trajectory import validate_trajectory, validate_model_bounds
from nero_agent.stream_guard import StreamGuard
from nero_agent.ros_backend import RosBackend
from nero_agent.chooser import OpenRouterChooser, decode_action_response

ROOT = Path(__file__).resolve().parents[1]


def config():
    return json.loads((ROOT / 'examples/nero-agent.mock.json').read_text())


class AgentTests(unittest.TestCase):
    def test_direct_pose_chooser_bypasses_llm_after_one_move(self):
        chooser = DirectPoseChooser('table_photo_center')
        self.assertEqual(chooser.choose({})['pose'], 'table_photo_center')
        self.assertEqual(chooser.choose({})['action'], 'finish')

    def test_llm_parser_accepts_object_and_content_block_variants(self):
        action = {'action': 'move_to_named_pose', 'pose': 'inspection', 'reason': 'Inspect'}
        for content in (action, [{'type': 'text', 'text': json.dumps(action)}], json.dumps(action)):
            body = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': content}}]})
            self.assertEqual(decode_action_response(body), action)

    def test_llm_parser_reports_response_shape(self):
        with self.assertRaisesRegex(AgentError, 'finish_reason=length'):
            decode_action_response(json.dumps({'choices': [{'finish_reason': 'length', 'message': {}}]}))

    def test_ros_executor_uses_private_context_and_closes_before_context(self):
        events = []
        context, node, executor = MagicMock(), MagicMock(), MagicMock()
        executor.shutdown.side_effect = lambda **kw: events.append('executor')
        node.destroy_node.side_effect = lambda: events.append('node')
        context.shutdown.side_effect = lambda: events.append('context')
        factory = MagicMock(return_value=executor)
        ros = NS(context=NS(Context=lambda: context), init=MagicMock(),
                 create_node=MagicMock(return_value=node))
        modules = {'rclpy': ros, 'rclpy.action': NS(ActionClient=MagicMock()),
                   'rclpy.executors': NS(SingleThreadedExecutor=factory),
                   'sensor_msgs.msg': NS(JointState=object),
                   'std_msgs.msg': NS(Empty=object, String=object),
                   'std_srvs.srv': NS(Trigger=object, SetBool=object),
                   'moveit_msgs.srv': NS(GetMotionPlan=object, ApplyPlanningScene=object, GetPlanningScene=object),
                   'moveit_msgs.action': NS(ExecuteTrajectory=object)}
        with patch.dict('sys.modules', modules):
            backend = RosBackend(Settings.parse(config()), require_ready=False)
            factory.assert_called_once_with(context=context)
            executor.add_node.assert_called_once_with(node)
            backend._spin()
            executor.spin_once.assert_called_once_with(timeout_sec=0.01)
            backend.close()
        self.assertEqual(events, ['executor', 'node', 'context'])

    def test_cleanup_ignores_repeat_interrupt_and_escalates_stuck_child(self):
        import signal
        import subprocess
        from nero_agent.bringup import stop_processes
        process = MagicMock(pid=12345)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired('ros', 8), None]
        with patch('nero_agent.bringup.signal.signal', return_value='old') as handler, \
                patch('nero_agent.bringup.os.killpg') as kill:
            stop_processes([process])
        self.assertEqual(handler.call_args_list[0].args, (signal.SIGINT, signal.SIG_IGN))
        self.assertEqual(handler.call_args_list[-1].args, (signal.SIGINT, 'old'))
        self.assertEqual([call.args for call in kill.call_args_list],
                         [(12345, signal.SIGINT), (12345, signal.SIGTERM)])

    def test_llm_request_limits_actions_and_receives_measured_context(self):
        captured = []
        action = dict(action='move_to_named_pose', pose='inspection', reason='Requested inspection')
        body = json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(action)}}]}).encode()
        def transport(request, timeout):
            captured.append(json.loads(request.data))
            return io.BytesIO(body)
        chooser = OpenRouterChooser.__new__(OpenRouterChooser)
        chooser.key, chooser.model, chooser.transport = 'test-key', 'test-model', transport
        context = dict(instruction='inspect', state={'source': 'hardware_feedback', 'joints_rad': [0.] * 7},
                       named_poses={'start': [0.] * 7, 'inspection': [0.02] + [0.] * 6}, history=[])
        self.assertEqual(chooser.choose(context), action)
        payload = captured[0]
        self.assertEqual(json.loads(payload['messages'][1]['content']), context)
        self.assertFalse(payload['provider']['allow_fallbacks'])
        schema = payload['response_format']['json_schema']['schema']
        self.assertEqual(schema['properties']['pose']['anyOf'][1]['enum'], ['start', 'inspection'])

    def test_llm_incomplete_reply_is_rejected(self):
        chooser = OpenRouterChooser.__new__(OpenRouterChooser)
        chooser.key, chooser.model = 'test-key', 'test-model'
        chooser.transport = lambda *a, **k: io.BytesIO(json.dumps({'choices': [
            {'finish_reason': 'length', 'message': {'content': '{}'}}]}).encode())
        with self.assertRaisesRegex(AgentError, 'incomplete'):
            chooser.choose({'named_poses': {'start': [0.] * 7}})

    def test_hardware_execution_publishes_heartbeat_while_spinning(self):
        backend = RosBackend.__new__(RosBackend)
        published, spins = [], []
        backend.hardware, backend.motion_pending, backend.heartbeat_at = True, True, 0.0
        backend.heartbeat = NS(publish=published.append)
        backend.node = object()
        backend.callback_executor = NS(spin_once=lambda timeout_sec: spins.append(backend.node))
        with patch.dict('sys.modules', {'std_msgs.msg': NS(Empty=lambda: 'heartbeat')}):
            backend._spin()
            backend._spin()
        self.assertEqual(published, ['heartbeat'])
        self.assertEqual(spins, [backend.node, backend.node])

    def test_scripted_roundtrip_uses_new_state_for_each_decision(self):
        settings = Settings.parse(config())
        backend = OfflineBackend(settings)
        contexts = []
        scripted = ScriptedChooser()
        def choose(context):
            contexts.append(deepcopy(context))
            return scripted.choose(context)
        events = []
        result = run_loop(backend, NS(choose=choose), settings, 'test', True, events.append, lambda _: False)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['returned_to_start'])
        self.assertEqual(contexts[0]['state']['joints_rad'][0], 0)
        self.assertEqual(contexts[1]['state']['joints_rad'][0], 0.02)
        self.assertEqual(contexts[2]['state']['joints_rad'][0], 0)
        self.assertEqual(contexts[1]['history'][0]['status'], 'executed')

    def test_plan_only_does_not_advance_measured_state_or_request_next_decision(self):
        settings = Settings.parse(config())
        backend = OfflineBackend(settings)
        with patch.object(backend, 'execute') as execute:
            result = run_loop(backend, ScriptedChooser(), settings, 'test', False, lambda _: None, lambda _: True)
        execute.assert_not_called()
        self.assertEqual(result['status'], 'planned_only')
        self.assertEqual(backend.q, (0,) * 7)

    def test_cancelled_confirmation_prevents_physical_command(self):
        settings = Settings.parse(config())
        backend = OfflineBackend(settings)
        backend.hardware = True
        with patch.object(backend, 'execute') as execute:
            with self.assertRaisesRegex(AgentError, 'cancelled'):
                run_loop(backend, ScriptedChooser(), settings, 'test', True, lambda _: None, lambda _: False)
        execute.assert_not_called()

    def test_failed_execution_is_not_followed_by_return(self):
        settings = Settings.parse(config())
        backend = OfflineBackend(settings)
        with patch.object(backend, 'execute', side_effect=AgentError('tracking failed')) as execute:
            with self.assertRaisesRegex(AgentError, 'tracking failed'):
                run_loop(backend, ScriptedChooser(), settings, 'test', True, lambda _: None, lambda _: True)
        self.assertEqual(execute.call_count, 1)

    def test_unreached_goal_is_not_success(self):
        settings = Settings.parse(config())
        backend = OfflineBackend(settings)
        with patch.object(backend, 'execute', return_value=backend.state()):
            with self.assertRaisesRegex(AgentError, 'did not reach'):
                run_loop(backend, ScriptedChooser(), settings, 'test', True, lambda _: None, lambda _: True)

    def test_invalid_actions_and_configurations(self):
        for action in ({'action': 'move_j', 'pose': 'inspection', 'reason': ''},
                       {'action': 'move_to_named_pose', 'pose': 'invented', 'reason': ''},
                       {'action': 'stop', 'pose': 'inspection', 'reason': ''}):
            with self.subTest(action=action), self.assertRaises(AgentError):
                validate_action(action, ['inspection'])
        for text in ('{"x":1,"x":2}', '{"x":NaN}'):
            with self.assertRaises(AgentError):
                strict_json(text)
        c = config()
        c['mode'] = 'hardware'
        with self.assertRaisesRegex(AgentError, 'reviewed_hardware'):
            Settings.parse(c)
        c = config()
        c['named_poses']['inspection']['delta_from_start_rad'][0] = 0.31
        with self.assertRaisesRegex(AgentError, 'excursion'):
            Settings.parse(c).resolve([0] * 7)

    def test_budget_limits_observation_loop(self):
        settings = Settings.parse(config())
        chooser = NS(choose=lambda _: dict(action='get_state', pose=None, reason='observe'))
        with self.assertRaisesRegex(AgentError, 'budget'):
            run_loop(OfflineBackend(settings), chooser, settings, 'test', True, lambda _: None, lambda _: True, 2)

    def test_cli_offline_has_no_ros_sdk_or_api_dependency(self):
        with tempfile.TemporaryDirectory() as d, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            out = Path(d) / 'result.json'
            with patch.dict('sys.modules', {'rclpy': None, 'pyAgxArm': None, 'mujoco': None}):
                self.assertEqual(main(['run', '--offline-demo', '--output', str(out)]), 0)
            result = json.loads(out.read_text())
            self.assertEqual(result['status'], 'offline_demo_passed')
            self.assertEqual(result['backend'], 'offline_test_double')
            self.assertFalse(result['hardware_motion_may_have_occurred'])
            self.assertEqual(main(['run', '--offline-demo', '--execute', '--output', str(out)]), 2)
            self.assertEqual(main(['run', '--offline-demo', '--config', str(out), '--output', str(out)]), 2)
            self.assertEqual(json.loads(out.read_text()), result)

    def test_report_failure_before_execution_prevents_command(self):
        settings = Settings.parse(config())
        backend = OfflineBackend(settings)
        def record(event):
            if event['event'] == 'execution_pending':
                raise OSError('disk full')
        with patch.object(backend, 'execute') as execute:
            with self.assertRaises(OSError):
                run_loop(backend, ScriptedChooser(), settings, 'test', True, record, lambda _: True)
        execute.assert_not_called()


class TrajectoryTests(unittest.TestCase):
    def test_model_limits_reject_captured_hardware_state_and_invalid_goal(self):
        description = '<robot>' + ''.join(
            '<joint name="%s" type="revolute"><limit lower="-1.74" upper="1.74"/></joint>' % name
            for name in JOINTS) + '</robot>'
        validate_model_bounds(description, [0.] * 7, [0.02] * 7)
        outside = [0., 1.8328051541042854, 0., 0., 0., 0., 0.]
        with self.assertRaisesRegex(AgentError, 'Measured start joint2 1.832805'):
            validate_model_bounds(description, outside, outside)
        with self.assertRaisesRegex(AgentError, 'Requested goal joint2'):
            validate_model_bounds(description, [0.] * 7, outside)
        with self.assertRaisesRegex(AgentError, 'Cannot validate loaded robot model'):
            validate_model_bounds('<robot/>', [0.] * 7, [0.] * 7)

    def test_clamped_endpoint_error_identifies_joint_and_values(self):
        trajectory = self.trajectory()
        trajectory.points[-1].positions[1] = -0.09
        with self.assertRaisesRegex(AgentError, 'joint2 planned -0.090000 rad, requested 0.000000'):
            validate_trajectory(trajectory, [0.] * 7, [0.02] + [0.] * 6,
                                [0.] * 7, Settings.parse(config()))

    def test_ros_numeric_arrays_pass_with_all_bounds_preserved(self):
        settings = Settings.parse(config())
        start, goal = [0.] * 7, [0.02] + [0.] * 6
        trajectory = self.trajectory()
        for point in trajectory.points:
            for field in ('positions', 'velocities', 'accelerations'):
                setattr(point, field, array('d', getattr(point, field)))
        self.assertEqual(validate_trajectory(trajectory, start, goal, start, settings)['duration_s'], 2)
        for field, invalid in (('positions', [0.] * 6), ('positions', [float('nan')] * 7),
                               ('positions', [float('inf')] * 7), ('positions', [0.5] * 7),
                               ('velocities', [0.5] * 7), ('accelerations', [1.] * 7)):
            broken = deepcopy(trajectory)
            setattr(broken.points[1], field, array('d', invalid))
            with self.subTest(field=field, value=invalid), self.assertRaises(AgentError):
                validate_trajectory(broken, start, goal, start, settings)

    def trajectory(self):
        return NS(joint_names=list(JOINTS), points=[
            NS(positions=[0.] * 7, velocities=[0.] * 7, accelerations=[0.] * 7,
               time_from_start=NS(sec=0, nanosec=0)),
            NS(positions=[0.02] + [0.] * 6, velocities=[0.] * 7, accelerations=[0.] * 7,
               time_from_start=NS(sec=2, nanosec=0))])

    def test_rejects_malformed_or_out_of_bounds_moveit_output(self):
        settings = Settings.parse(config())
        start, goal = [0.] * 7, [0.02] + [0.] * 6
        self.assertEqual(validate_trajectory(self.trajectory(), start, goal, start, settings)['duration_s'], 2)
        for change in ('joints', 'velocity', 'acceleration', 'time', 'excursion', 'nan'):
            t = self.trajectory()
            if change == 'joints': t.joint_names.reverse()
            if change == 'velocity': t.points[1].velocities[0] = 0.5
            if change == 'acceleration': t.points[1].accelerations[0] = 1
            if change == 'time': t.points[1].time_from_start.sec = 0
            if change == 'excursion': t.points[1].positions[0] = 0.5
            if change == 'nan': t.points[1].positions[0] = float('nan')
            with self.subTest(change=change), self.assertRaises(AgentError):
                validate_trajectory(t, start, goal, start, settings)

    def test_excursion_reports_largest_path_deviation_not_only_endpoint(self):
        settings = Settings.parse(config())
        t = self.trajectory()
        middle = deepcopy(t.points[1])
        middle.time_from_start.sec = 1
        middle.positions = [0., 0., 0., -.45, 0., 0., 0.]
        t.points.insert(1, middle)
        with self.assertRaisesRegex(AgentError, 'joint4 displacement 0.450000') as raised:
            validate_trajectory(t, [0.] * 7, t.points[-1].positions, [0.] * 7, settings)
        self.assertIn('limit 0.300000 rad, excess 0.150000 rad', str(raised.exception))
        self.assertIn('time 1.000000 s', str(raised.exception))

    def test_timeout_reports_full_path_after_first_failing_timestamp(self):
        t = self.trajectory()
        t.points[1].time_from_start.sec = 31
        last = deepcopy(t.points[1])
        last.time_from_start.sec = 45
        last.positions[3] = -.45
        t.points.append(last)
        with self.assertRaisesRegex(AgentError, 'exceeds execution timeout') as raised:
            validate_trajectory(t, [0.] * 7, last.positions, [0.] * 7, Settings.parse(config()))
        message = str(raised.exception)
        for detail in ('point index 1', 'timestamp 31.000000 s', 'timeout 30.000000 s',
                       'duration if timing is valid) 45.000000 s',
                       'joint4 displacement 0.450000 rad', 'excess 0.150000 rad'):
            self.assertIn(detail, message)

    def test_malformed_timing_has_distinct_reason_and_context(self):
        for stamp, reason in ((0, 'do not strictly increase'), (-1, 'negative'),
                              (float('nan'), 'nonfinite'), (float('inf'), 'nonfinite')):
            with self.subTest(stamp=stamp):
                t = self.trajectory()
                t.points[1].time_from_start.sec = stamp
                with self.assertRaisesRegex(AgentError, reason) as raised:
                    validate_trajectory(t, [0.] * 7, t.points[1].positions,
                                        [0.] * 7, Settings.parse(config()))
                self.assertIn('point index 1', str(raised.exception))
                self.assertIn('previous timestamp 0.000000 s', str(raised.exception))
                self.assertIn('joint1 displacement 0.020000 rad', str(raised.exception))

    def test_execution_timeout_boundary_is_accepted(self):
        t = self.trajectory()
        t.points[1].time_from_start.sec = 30
        result = validate_trajectory(t, [0.] * 7, t.points[1].positions,
                                     [0.] * 7, Settings.parse(config()))
        self.assertEqual(result['duration_s'], 30.)

    def test_plan_accepts_new_excursion_boundary_and_rejects_beyond(self):
        settings = Settings.parse(config())
        t = self.trajectory()
        t.points[1].positions[0] = -.30
        t.points[1].time_from_start.sec = 20
        result = validate_trajectory(t, [0.] * 7, t.points[-1].positions, [0.] * 7, settings)
        self.assertEqual(result['peak_excursion_rad'], .30)
        t.points[1].positions[0] = -.301
        with self.assertRaisesRegex(AgentError, 'joint1 displacement 0.301000'):
            validate_trajectory(t, [0.] * 7, t.points[-1].positions, [0.] * 7, settings)


class GuardTests(unittest.TestCase):
    def test_expanded_stream_envelope_keeps_tracking_and_feedback_bounds(self):
        guard = StreamGuard([0.] * 7, .04, 0)
        for i in range(301):
            q = [i * .001] + [0.] * 6
            guard.command(q, q, 100. + i * .1, 100. + i * .1, i * .1)
        q = [.301] + [0.] * 6
        with self.assertRaisesRegex(AgentError, 'excursion bounds: joint1'):
            guard.command(q, q, 130.1, 130.1, 30.1)
        guard = StreamGuard([0.] * 7, .04, 0)
        guard.check_feedback([.31] + [0.] * 6, [0.] * 7, .04, .01)
        with self.assertRaisesRegex(AgentError, 'envelope: joint1'):
            guard.check_feedback([.311] + [0.] * 6, [0.] * 7, .04, .02)
        with self.assertRaisesRegex(AgentError, 'tracking bounds'):
            guard.command([.20] + [0.] * 6, [.18] + [0.] * 6, 100., 100., .03)

    def test_velocity_trip_retains_offending_feedback_and_command(self):
        guard = StreamGuard([0.] * 7, .04, 0)
        guard.command([0.] * 7, [0.] * 7, 100., 100., .01)
        speeds = [0.] * 7
        speeds[3] = -.125
        guard.check_feedback([0.] * 7, speeds, .04, .01,
                             velocity_timestamps=[99.94] * 7, wall_now=99.99)
        with self.assertRaisesRegex(AgentError, 'joint4 -0.125000') as raised:
            guard.check_feedback([0.] * 7, speeds, .04, .02,
                                 velocity_timestamps=[99.95] * 7, wall_now=100.)
        history = raised.exception.history
        self.assertEqual(history[0]['event'], 'command_received')
        self.assertEqual(history[-1]['velocities_rad_s'][3], -.125)
        for age in history[-1]['motor_velocity_ages_s']:
            self.assertAlmostEqual(age, .05)
        self.assertEqual(history[-1]['last_command_rad'], (0.,) * 7)

    def test_feedback_history_compares_motor_velocity_to_position_packets(self):
        guard = StreamGuard([0.] * 7, .04, 0)
        guard.check_feedback([0.] * 7, [0.] * 7, .04, .01,
                             position_timestamps=[1.] * 4)
        guard.check_feedback([.001] + [0.] * 6, [.09] + [0.] * 6, .04, .02,
                             position_timestamps=[1.01] * 4)
        event = guard.history[-1]
        self.assertAlmostEqual(event['joint_finite_difference_velocities_rad_s'][0], .1)
        self.assertAlmostEqual(event['motor_minus_position_velocity_rad_s'][0], -.01)

    def test_motor_limit_can_be_explicitly_raised_but_position_limit_remains(self):
        guard = StreamGuard([0.] * 7, .04, 0, motor_velocity_limit=.11)
        guard.check_feedback([0.] * 7, [.105] + [0.] * 6, .04, .01,
                             position_timestamps=[1.] * 4)
        with self.assertRaisesRegex(AgentError, 'Position-derived velocity'):
            guard.check_feedback([.002] + [0.] * 6, [.105] + [0.] * 6, .04, .02,
                                 position_timestamps=[1.01] * 4)

    def test_position_velocity_limit_accepts_up_to_point_fifteen(self):
        for speed in (-.151, -.15, -.106672, .106672, .15, .151):
            with self.subTest(speed=speed):
                guard = StreamGuard([0.] * 7, .04, 0)
                guard.check_feedback([0.] * 7, [0.] * 7, .04, .01,
                                     position_timestamps=[1.] * 4)
                # Exact binary time interval avoids rounding at the boundary.
                q = [0.] * 4 + [speed * .125] + [0.] * 2
                if abs(speed) <= .15:
                    guard.check_feedback(q, [0.] * 7, .04, .135,
                                         position_timestamps=[1.125] * 4)
                else:
                    with self.assertRaisesRegex(AgentError, 'Position-derived velocity exceeded 0.150 rad/s: joint5'):
                        guard.check_feedback(q, [0.] * 7, .04, .135,
                                             position_timestamps=[1.125] * 4)

    def test_watchdogs_and_gripper_motion_abort(self):
        for now, width, velocity in ((0.3, .04, 0), (0.1, .05, 0), (0.1, .04, .201)):
            guard = StreamGuard([0] * 7, .04, 0)
            with self.assertRaises(AgentError):
                guard.check_feedback([0] * 7, [velocity] * 7, width, now)
        guard = StreamGuard([0] * 7, .04, 0)
        guard.command_at = 1
        with self.assertRaisesRegex(AgentError, 'heartbeat'):
            guard.check_feedback([0] * 7, [0] * 7, .04, 1)

    def test_motor_filter_rejects_isolated_spike_from_recorded_run(self):
        guard = StreamGuard([0.] * 7, .04, 0, motor_velocity_limit=.15)
        for i, speed in enumerate((-.022, -.013, -.019, -.002, -.022, -.157, -.022)):
            guard.check_feedback([0.] * 7, [0., 0., speed, 0., 0., 0., 0.],
                                 .04, i * .01, velocity_timestamps=[100 + i * .01] * 7)

    def test_motor_filter_stops_sustained_or_alternating_overspeed(self):
        for speeds in ((.16, .16), (.16, -.16), (.16, .01, .16)):
            guard = StreamGuard([0.] * 7, .04, 0, motor_velocity_limit=.15)
            for i, speed in enumerate(speeds[:-1]):
                guard.check_feedback([0.] * 7, [speed] * 7, .04, i * .01)
            with self.assertRaisesRegex(AgentError, '3-sample median'):
                guard.check_feedback([0.] * 7, [speeds[-1]] * 7, .04, (len(speeds)-1) * .01)

    def test_motor_filter_does_not_recount_cache_or_reuse_old_spikes(self):
        guard = StreamGuard([0.] * 7, .04, 0, motor_velocity_limit=.15)
        for now, stamp in ((0., 100.), (.01, 100.), (.02, 100.), (.04, 100.04)):
            guard.check_feedback([0.] * 7, [.16] * 7, .04, now,
                                 velocity_timestamps=[stamp] * 7)
        with self.assertRaisesRegex(AgentError, '3-sample median'):
            guard.check_feedback([0.] * 7, [.16] * 7, .04, .05,
                                 velocity_timestamps=[100.05] * 7)

    def test_motor_filter_hard_limit_is_immediate(self):
        guard = StreamGuard([0.] * 7, .04, 0, motor_velocity_limit=.15)
        with self.assertRaisesRegex(AgentError, 'instantaneous hard limit 0.300'):
            guard.check_feedback([0.] * 7, [-.301] * 7, .04, .01)

    def test_initial_command_cannot_jump_from_mock_zero_to_hardware(self):
        guard = StreamGuard([1] * 7, .04, 0)
        with self.assertRaises(AgentError):
            guard.command([0] * 7, [1] * 7, 100, 100, 0)

    def test_stream_speed_and_old_commands_are_rejected(self):
        guard = StreamGuard([0] * 7, .04, 0)
        guard.command([0] * 7, [0] * 7, 100, 100, 0)
        guard.command([.0005] * 7, [0] * 7, 100.01, 100.01, .01)
        with self.assertRaisesRegex(AgentError, 'velocity'):
            guard.command([.01] * 7, [0] * 7, 100.02, 100.02, .02)
        with self.assertRaisesRegex(AgentError, 'stale'):
            guard.command([0] * 7, [0] * 7, 100, 101, 1)


class RosFeedbackTests(unittest.TestCase):
    def backend(self):
        backend = RosBackend.__new__(RosBackend)
        backend.source = 'hardware_feedback'
        backend.node = NS(get_clock=lambda: NS(now=lambda: NS(nanoseconds=100_000_000_000)))
        backend.sample = None
        backend.sample_received = 0
        backend.sample_error = None
        backend.driver_fault = None
        return backend

    def sample(self):
        return NS(name=list(JOINTS) + ['gripper'], position=[0.] * 7 + [.04], velocity=[0.] * 8,
                  header=NS(stamp=NS(sec=100, nanosec=0)))

    def test_requires_all_joints_and_gripper_with_fresh_timestamp(self):
        for change in ('missing', 'duplicate', 'stale', 'nan'):
            b, s = self.backend(), self.sample()
            if change == 'missing': s.name.pop(); s.position.pop()
            if change == 'duplicate': s.name[-1] = 'joint1'
            if change == 'stale': s.header.stamp.sec = 99
            if change == 'nan': s.position[0] = float('nan')
            b._sample(s)
            with self.subTest(change=change), self.assertRaises(AgentError):
                b._fresh()

    def test_driver_fault_takes_precedence_over_recent_positions(self):
        b = self.backend()
        b._sample(self.sample())
        self.assertEqual(b._fresh()['gripper_width_m'], .04)
        b._fault(NS(data='overspeed'))
        with self.assertRaisesRegex(AgentError, 'overspeed'):
            b._fresh()


class RosExecutionTests(unittest.TestCase):
    def backend(self):
        b = RosBackend.__new__(RosBackend)
        b.settings = Settings.parse(config())
        b.hardware = False
        b.motion_pending = False
        b.goal_handle = None
        b.pending_goal = None
        start = {'joints_rad': [0.] * 7, 'gripper_width_m': .04, 'velocities_rad_s': [0.] * 7}
        b.plans = {'test': (object(), start, [.02] + [0.] * 6, 'unchanged')}
        b.state = lambda: deepcopy(start)
        b._scene_digest = lambda: 'unchanged'
        b.executor = NS(wait_for_server=lambda **k: True, send_goal_async=lambda req: 'goal_future')
        b.stop_calls = []
        b.stop = lambda: b.stop_calls.append(True)
        return b

    def imports(self):
        return patch.dict('sys.modules', {
            'std_srvs.srv': NS(SetBool=NS(Request=NS)),
            'moveit_msgs.action': NS(ExecuteTrajectory=NS(Goal=NS)),
            'moveit_msgs.msg': NS(MoveItErrorCodes=NS(SUCCESS=1)),
            'action_msgs.msg': NS(GoalStatus=NS(STATUS_SUCCEEDED=4)),
        })

    def test_changed_arm_gripper_or_scene_cannot_execute_cached_plan(self):
        for change in ('arm', 'gripper', 'scene'):
            b = self.backend()
            state = b.state()
            if change == 'arm': state['joints_rad'][0] = .02
            if change == 'gripper': state['gripper_width_m'] = .08
            if change == 'scene': b._scene_digest = lambda: 'changed'
            b.state = lambda: state
            with self.imports(), self.assertRaisesRegex(AgentError, 'changed'):
                b.execute({'token': 'test'})
            self.assertFalse(b.motion_pending)
            self.assertEqual(b.stop_calls, [])

    def test_action_rejection_triggers_stop_and_consumes_plan(self):
        b = self.backend()
        b._wait = lambda *a, **k: NS(accepted=False)
        with self.imports(), self.assertRaisesRegex(AgentError, 'rejected'):
            b.execute({'token': 'test'})
        self.assertEqual(b.stop_calls, [True])
        self.assertNotIn('test', b.plans)

    def test_unverified_abort_blocks_execution_before_any_backend_action(self):
        b = self.backend()
        b.hardware = True
        b.state = MagicMock()
        b.executor.send_goal_async = MagicMock()
        with self.assertRaisesRegex(AgentError, 'powered controlled abort'):
            b.execute({'token': 'test'})
        b.state.assert_not_called()
        b.executor.send_goal_async.assert_not_called()
        self.assertFalse(b.motion_pending)
        self.assertEqual(b.stop_calls, [])

    @patch('nero_agent.abort_policy.require_verified_controlled_abort')
    def test_hardware_setup_requires_new_unchanged_feedback_before_sending(self, capability):
        for failure in ('stale', 'moved', 'gripper'):
            b = self.backend()
            b.hardware, b.gate = True, object()
            initial = b.state()
            b.sample = 'cached before blocking setup'
            b._call = lambda *a, **k: NS(success=True)
            def refreshed():
                self.assertIsNone(b.sample)
                if failure == 'stale':
                    raise AgentError('No fresh complete joint/gripper feedback')
                changed = deepcopy(initial)
                if failure == 'moved': changed['joints_rad'][0] += .02
                else: changed['gripper_width_m'] += .01
                return changed
            count = [0]
            def state():
                count[0] += 1
                return initial if count[0] == 1 else refreshed()
            b.state = state
            b.executor.send_goal_async = MagicMock()
            with self.subTest(failure=failure), self.imports(), self.assertRaises(AgentError):
                b.execute({'token': 'test'})
            b.executor.send_goal_async.assert_not_called()
            self.assertEqual(b.stop_calls, [True])

    @patch('nero_agent.abort_policy.require_verified_controlled_abort')
    def test_hardware_setup_refreshes_feedback_before_action_submission(self, capability):
        b = self.backend()
        b.hardware, b.gate = True, object()
        initial = b.state()
        events = []
        b.sample = 'old sample'
        def gate(*args):
            events.append('setup')
            return NS(success=True)
        def state():
            if events:
                self.assertIsNone(b.sample)
                events.append('fresh feedback')
            return initial
        b._call, b.state = gate, state
        b.executor.send_goal_async = lambda req: events.append('send')
        b._wait = lambda *a, **k: NS(accepted=False)
        with self.imports(), self.assertRaisesRegex(AgentError, 'rejected'):
            b.execute({'token': 'test'})
        self.assertEqual(events, ['setup', 'fresh feedback', 'send'])

    @patch('nero_agent.abort_policy.require_verified_controlled_abort')
    def test_hardware_execution_replans_from_final_measured_state(self, capability):
        b = self.backend()
        b.hardware, b.gate = True, object()
        initial = b.state()
        refreshed = deepcopy(initial)
        refreshed['joints_rad'][0] += .001
        states = [initial, refreshed]
        b.state = lambda: states.pop(0)
        b.sample = 'old sample'
        b._call = lambda *a, **k: NS(success=True)
        replanned = MagicMock(return_value=(object(), 'unchanged', {}))
        b._plan_from_state = replanned
        b.executor.send_goal_async = MagicMock()
        b._wait = lambda *a, **k: NS(accepted=False)
        with self.imports(), self.assertRaisesRegex(AgentError, 'rejected'):
            b.execute({'token': 'test'})
        replanned.assert_called_once_with([.02] + [0.] * 6, refreshed)
        b.executor.send_goal_async.assert_called_once()

    def test_action_success_without_measured_arrival_is_rejected(self):
        b = self.backend()
        b._wait = lambda future, *a, **k: (
            NS(accepted=True, get_result_async=lambda: 'result') if future == 'goal_future' else
            NS(status=4, result=NS(error_code=NS(val=1))))
        b._spin = lambda: None
        b._fresh = b.state
        # Advance past the measured arrival deadline without sleeping in tests.
        with self.imports(), patch('nero_agent.ros_backend.time.monotonic', side_effect=[0, 4]):
            with self.assertRaisesRegex(AgentError, 'did not settle'):
                b.execute({'token': 'test'})
        self.assertEqual(b.stop_calls, [True])

    def test_segment_waits_for_stricter_speed_and_sustained_standstill(self):
        import itertools
        b = self.backend()
        b.executing_segment = True
        b._wait = lambda future, *a, **k: (
            NS(accepted=True, get_result_async=lambda: 'result') if future == 'goal_future' else
            NS(status=4, result=NS(error_code=NS(val=1))))
        b._spin = lambda: None
        samples = []
        def measured():
            samples.append(True)
            return {'joints_rad': [.02]+[0.]*6, 'gripper_width_m': .04,
                    'velocities_rad_s': [.004 if len(samples) <= 3 else 0.]+[0.]*6}
        b._fresh = measured
        clock = itertools.count(1., .1)
        with self.imports(), patch('nero_agent.ros_backend.time.monotonic', side_effect=lambda: next(clock)):
            result = b.execute({'token': 'test'})
        self.assertGreaterEqual(len(samples), 6)
        self.assertEqual(result['velocities_rad_s'], [0.]*7)
        self.assertEqual(b.stop_calls, [])

    @patch('nero_agent.abort_policy.require_verified_controlled_abort')
    def test_segment_setup_drift_rejects_before_action_without_replanning(self, capability):
        b = self.backend()
        b.hardware, b.gate, b.executing_segment = True, object(), True
        initial = b.state()
        moved = deepcopy(initial)
        moved['joints_rad'][0] = .0006
        states = iter([initial, moved])
        b.state = lambda: next(states)
        b._call = lambda *a, **k: NS(success=True)
        b._plan_from_state = MagicMock()
        b.executor.send_goal_async = MagicMock()
        with self.imports(), self.assertRaisesRegex(AgentError, 'changed during hardware setup'):
            b.execute({'token': 'test'})
        b.executor.send_goal_async.assert_not_called()
        b._plan_from_state.assert_not_called()
        self.assertEqual(b.stop_calls, [True])
