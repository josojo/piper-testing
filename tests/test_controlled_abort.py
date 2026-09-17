from types import SimpleNamespace as NS
from unittest.mock import MagicMock
import unittest

from nero_agent.controlled_abort import ControlledAbort


class AbortTests(unittest.TestCase):
    def setup_abort(self):
        self.now = 0.
        self.events = []
        self.q, self.v = [0.] * 7, [0.] * 7
        def read(require_enabled):
            self.assertTrue(require_enabled)
            self.events.append('read')
            return NS(joints_rad=self.q[:], velocities_rad_s=self.v[:], enabled=[True] * 7,
                      feedback_timestamps=[100 + self.now] * 7)
        def move(q):
            self.events.append('move_j')
        self.hardware = NS(read=MagicMock(side_effect=read), robot=NS(move_j=MagicMock(side_effect=move)))
        self.future = NS(done=lambda: True, result=lambda: NS(return_code=0))
        def cancel():
            self.events.append('cancel')
            return self.future
        self.abort = ControlledAbort(self.hardware, lambda: self.events.append('block'), cancel,
                                     clock=lambda: self.now)
        self.abort.start('test')
        return self.abort

    def test_order_and_observed_sustained_hold(self):
        a = self.setup_abort()
        a.tick()
        self.assertEqual(self.events, ['block', 'cancel', 'read', 'move_j'])
        for t in (.1, .5, 1.2):
            self.now = t; a.tick()
        self.assertEqual(a.phase, 'holding')
        self.assertEqual(a.result()['feedback_samples'], 3)
        self.assertFalse(a.result()['physically_validated'])
        self.assertEqual(self.hardware.robot.move_j.call_count, 1)

    def test_timing_records_blocking_move_j_separately(self):
        a = self.setup_abort()
        self.now = .02
        def move(q):
            self.now += .1
        self.hardware.robot.move_j.side_effect = move
        a.tick()
        timing = a.result()['timings']
        self.assertEqual(timing['abort_requested_monotonic_s'], 0.)
        self.assertAlmostEqual(timing['move_j_returned_monotonic_s'] - timing['move_j_started_monotonic_s'], .1)

    def test_abort_is_idempotent_and_does_not_chase_position(self):
        a = self.setup_abort(); a.tick()
        self.q[0] = .001
        a.start('again'); self.now = .1; a.tick()
        self.assertEqual(a.target, (0.,) * 7)
        self.hardware.robot.move_j.assert_called_once_with([0.] * 7)
        self.assertEqual(self.events.count('block'), 1)

    def test_unavailable_cancel_does_not_skip_hold(self):
        a = self.setup_abort()
        self.future.result = MagicMock(side_effect=RuntimeError('cancel failed'))
        a.tick()
        self.assertEqual(a.phase, 'settling')
        self.assertIn('cancel failed', a.cancellation)
        self.hardware.robot.move_j.assert_called_once()

    def test_cancellation_timeout_is_bounded_then_attempts_hold(self):
        a = self.setup_abort(); self.future.done = lambda: False
        self.now = .1; a.tick()
        self.hardware.robot.move_j.assert_not_called()
        self.now = .21; a.tick()
        self.assertEqual(a.cancellation, 'timeout')
        self.hardware.robot.move_j.assert_called_once()

    def test_stale_or_disabled_feedback_prevents_hold_command(self):
        a = self.setup_abort()
        self.hardware.read.side_effect = RuntimeError('stale or disabled')
        a.tick()
        self.assertEqual(a.phase, 'failed')
        self.hardware.robot.move_j.assert_not_called()

    def test_send_failure_is_latched_no_retry(self):
        a = self.setup_abort()
        self.hardware.robot.move_j.side_effect = RuntimeError('send failed')
        a.tick(); a.tick(); a.start('retry')
        self.assertEqual(a.phase, 'failed')
        self.hardware.robot.move_j.assert_called_once()

    def test_excursion_and_no_settling_fail_without_second_command(self):
        for excursion in (True, False):
            a = self.setup_abort(); a.tick()
            if excursion: self.q[1] = .03
            else: self.v[1] = .15
            self.now = 5.1; a.tick()
            self.assertEqual(a.phase, 'failed')
            self.hardware.robot.move_j.assert_called_once()

    def test_motion_resets_standstill_dwell(self):
        a = self.setup_abort(); a.tick()
        self.now = .1; a.tick()
        self.v[0] = .02; self.now = .8; a.tick()
        self.v[0] = 0.; self.now = 1.; a.tick()
        self.now = 1.2; a.tick()
        self.assertEqual(a.phase, 'settling')
        self.now = 2.1; a.tick()
        self.assertEqual(a.phase, 'holding')
        self.v[0] = .02; self.now = 2.2; a.tick()
        self.assertEqual(a.phase, 'rechecking')
        self.v[0] = 0.; self.now = 2.3; a.tick()
        self.now = 3.4; a.tick()
        self.assertEqual(a.phase, 'holding')
        self.hardware.robot.move_j.assert_called_once()
        self.assertEqual(a.result()['recheck_count'], 1)

    def test_repeated_spikes_do_not_extend_recheck_deadline(self):
        a = self.setup_abort(); a.tick()
        self.now = .1; a.tick(); self.now = 1.2; a.tick()
        self.v[5] = -.035; self.now = 1.3; a.tick()
        deadline = a.recheck_deadline
        for t, speed in ((1.4, 0.), (2., -.035), (2.1, 0.), (2.8, -.035), (6.2, 0.)):
            self.now, self.v[5] = t, speed
            a.tick()
            self.assertEqual(a.phase, 'rechecking')
            self.assertEqual(a.recheck_deadline, deadline)
        self.now = 6.31; a.tick()
        self.assertEqual(a.phase, 'failed')
        self.assertIn('recheck did not settle', a.result()['failure'])
        a.start('retry')
        self.hardware.robot.move_j.assert_called_once()
        self.assertEqual(self.events.count('block'), 1)

    def test_recheck_preserves_excursion_and_feedback_failures(self):
        for failure in ('excursion', 'feedback'):
            a = self.setup_abort(); a.tick()
            self.now = .1; a.tick(); self.now = 1.2; a.tick()
            self.v[5] = -.035; self.now = 1.3; a.tick()
            if failure == 'excursion': self.q[0] = .021
            else: self.hardware.read.side_effect = RuntimeError('stale or disabled')
            self.now = 1.4; a.tick()
            self.assertEqual(a.phase, 'failed')
            self.hardware.robot.move_j.assert_called_once()

    def test_feedback_loss_while_holding_is_reported(self):
        a = self.setup_abort(); a.tick()
        self.now = .1; a.tick(); self.now = 1.2; a.tick()
        self.hardware.read.side_effect = RuntimeError('feedback lost')
        a.tick()
        self.assertEqual(a.phase, 'failed')
        self.assertEqual(a.result()['failure'], 'feedback lost')


class DriverAbortTests(unittest.TestCase):
    def test_driver_latches_queued_commands_and_separates_emergency_stop(self):
        from contextlib import contextmanager
        from unittest.mock import patch
        from nero_agent import driver
        state = NS(joints_rad=[0.] * 7, velocities_rad_s=[0.] * 7,
                   enabled=[True] * 7, feedback_timestamps=[100.] * 7)
        hardware = NS(robot=MagicMock(), read=MagicMock(return_value=state), stop=MagicMock())
        @contextmanager
        def connect():
            yield hardware
        class Node:
            def __init__(self, *args, **kwargs): pass
            def create_client(self, *args):
                return NS(service_is_ready=lambda: True,
                          call_async=lambda req: NS(done=lambda: True, result=lambda: NS(return_code=0)))
            def create_publisher(self, *args): return MagicMock()
            def create_subscription(self, *args): pass
            def create_service(self, *args): pass
            def create_timer(self, *args): pass
            def get_logger(self): return MagicMock()
            def destroy_node(self): pass
        def spin(node):
            node.guard = MagicMock()
            reply = node.stop_service(NS(), NS())
            self.assertTrue(reply.success)
            self.assertIsNone(node.guard)
            node.command(NS(name=['joint1'], position=[2.]))  # queued bad stream ignored
            hardware.robot.move_js.assert_not_called()
            node.abort.tick()
            hardware.robot.move_j.assert_called_once_with([0.] * 7)
            hardware.stop.assert_not_called()
            node.stop_service(NS(), NS())
            node.abort.tick()
            hardware.robot.move_j.assert_called_once()  # repeated stop cannot chase state
            reply = node.gate(NS(data=True), NS())
            self.assertFalse(reply.success)
            # Only an explicit emergency service call delivers the damping stop.
            reply = node.emergency_stop(NS(), NS())
            self.assertTrue(reply.success)
            hardware.stop.assert_called_once()
        modules = {'rclpy': NS(init=lambda: None, spin=spin, ok=lambda: True, shutdown=lambda: None),
                   'rclpy.node': NS(Node=Node), 'sensor_msgs.msg': NS(JointState=NS),
                   'std_msgs.msg': NS(Empty=NS, String=NS),
                   'std_srvs.srv': NS(SetBool=NS, Trigger=NS),
                   'action_msgs.srv': NS(CancelGoal=NS(Request=NS)),
                   'nero_experiment.hardware': NS(connect=connect, MAX_JOINT_SNAPSHOT_AGE_S=.055)}
        with patch.dict('sys.modules', modules), patch('sys.argv', ['driver']):
            driver.main()


    def test_diagnostic_mode_blocks_hardware_writes_and_records_bridge(self):
        from contextlib import contextmanager
        from unittest.mock import patch
        from nero_agent import driver
        state = NS(joints_rad=[0.] * 7, velocities_rad_s=[0.] * 7,
                   enabled=[True] * 7, feedback_timestamps=[100.] * 7)
        hardware = NS(robot=MagicMock(), read=MagicMock(return_value=state), stop=MagicMock())
        @contextmanager
        def connect():
            yield hardware
        class Node:
            def __init__(self, *args, **kwargs): pass
            def create_client(self, *args):
                return NS(service_is_ready=lambda: True,
                          call_async=lambda req: NS(done=lambda: True, result=lambda: NS(return_code=0)))
            def create_publisher(self, *args): return MagicMock()
            def create_subscription(self, *args): pass
            def create_service(self, *args): pass
            def create_timer(self, *args): pass
            def get_logger(self): return MagicMock()
            def destroy_node(self): pass
        def spin(node):
            import json
            for callback in (node.gate, node.commission_gate, node.stop_service, node.emergency_stop):
                self.assertFalse(callback(NS(data=True), NS()).success)
            node.command(NS(name=['joint1'], position=[2.]))
            hardware.robot.move_js.assert_not_called()
            hardware.robot.move_j.assert_not_called()
            hardware.stop.assert_not_called()
            self.assertIsNone(node.guard)
            self.assertTrue(node.diagnostic_start(NS(), NS()).success)
            self.assertFalse(node.diagnostic_start(NS(), NS()).success)
            self.assertFalse(node.abort_report(NS(), NS()).success)
            node.gripper = lambda: (0., 100.)
            # Minimal JointState mock deliberately lacks header: bridge error is recorded.
            node.tick()
            self.assertEqual(len(node.recorder.samples), 1)
            self.assertIn('bridge_error', node.recorder.samples[0])
            self.assertFalse(node.diagnostic_result(NS(), NS()).success)
            live = json.loads(node.abort_status(NS(), NS()).message)
            self.assertNotIn('history', live['commissioning'])
            node.diagnostic_end = 0
            full = node.abort_report(NS(), NS())
            self.assertTrue(full.success)
            self.assertIn('history', json.loads(full.message)['commissioning'])
            reply = node.diagnostic_result(NS(), NS())
            self.assertTrue(reply.success)
            self.assertEqual(json.loads(reply.message)['summary']['successful_reads'], 1)
            self.assertFalse(json.loads(reply.message)['motion_commands_sent'])
        modules = {'rclpy': NS(init=lambda: None, spin=spin, ok=lambda: True, shutdown=lambda: None),
                   'rclpy.node': NS(Node=Node), 'sensor_msgs.msg': NS(JointState=NS),
                   'std_msgs.msg': NS(Empty=NS, String=NS),
                   'std_srvs.srv': NS(SetBool=NS, Trigger=NS),
                   'action_msgs.srv': NS(CancelGoal=NS(Request=NS)),
                   'nero_experiment.hardware': NS(connect=connect, MAX_JOINT_SNAPSHOT_AGE_S=.055)}
        with patch.dict('sys.modules', modules), patch('sys.argv', ['driver', '--diagnose-feedback']):
            driver.main()


class BackendAbortTests(unittest.TestCase):
    def test_stop_blocks_via_driver_before_moveit_cancel_then_observes_hold(self):
        from unittest.mock import patch
        from nero_agent.ros_backend import RosBackend
        b = RosBackend.__new__(RosBackend)
        b.hardware, b.motion_pending = True, True
        b.source, b.pending_goal = 'hardware_feedback', None
        b.estop, b.abort_status = 'driver_stop', 'driver_status'
        b.fetch_abort_report = lambda observed: observed
        events = []
        b.goal_handle = NS(accepted=True, cancel_goal_async=lambda: events.append('cancel_moveit'))
        b._wait = lambda *args, **kwargs: None
        b._spin = lambda: None
        statuses = iter(['settling', 'holding'])
        def call(client, request, timeout):
            import json
            events.append(client)
            return NS(success=True, message=json.dumps({'status': next(statuses)})) if client == 'driver_status' else NS(success=True)
        b._call = call
        with patch.dict('sys.modules', {'std_srvs.srv': NS(Trigger=NS(Request=NS))}):
            result = b.stop()
        self.assertEqual(events, ['driver_stop', 'cancel_moveit', 'driver_status', 'driver_status'])
        self.assertEqual(result['status'], 'holding_observed')
        self.assertEqual(b.last_stop_result, {'status': 'holding'})
        self.assertFalse(b.motion_pending)


class AbortReportingTests(unittest.TestCase):
    def test_live_status_does_not_iterate_history(self):
        from nero_agent.commissioning import MovingAbortTrial
        class Trace:
            def __len__(self): return 1000
            def __iter__(self): raise AssertionError('Live status scanned trace')
        trial = MovingAbortTrial([0.] * 7, 0.)
        trial.history = Trace()
        self.assertEqual(trial.live_status()['feedback_samples'], 1000)
        self.assertNotIn('history', trial.live_status())

    def test_status_poll_rate_and_fresh_hold_report(self):
        from unittest.mock import patch
        from nero_agent.ros_backend import RosBackend
        backend = RosBackend.__new__(RosBackend)
        backend.abort_status, backend.abort_report = 'status', 'report'
        now = [1.]
        def spin(): now[0] += .01
        backend._spin = spin
        calls = []
        def call(client, request, timeout):
            calls.append((client, now[0]))
            return NS(success=True, message='{"status":"holding","commissioning":{"history":[]}}')
        backend._call = call
        with patch.dict('sys.modules', {'std_srvs.srv': NS(Trigger=NS(Request=NS))}), patch('nero_agent.ros_backend.time.monotonic', side_effect=lambda: now[0]):
            backend.poll_abort_status()
            backend.poll_abort_status()
            backend.fetch_abort_report({'status': 'holding'})
            backend.fetch_abort_report({'status': 'holding'})
        self.assertGreaterEqual(calls[1][1]-calls[0][1], .1)
        self.assertEqual([client for client, stamp in calls], ['status', 'status', 'report', 'report'])
