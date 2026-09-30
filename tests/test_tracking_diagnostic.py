from types import SimpleNamespace as NS
import unittest

from nero_agent.core import AgentError, JOINTS
from nero_agent.stream_guard import StreamGuard
from nero_agent.tracking_diagnostic import ControllerTrace


class TrackingTests(unittest.TestCase):
    def test_recorded_abort_hold_is_explained_but_remains_rejected(self):
        trace = ControllerTrace()
        actual = [0.]*7
        actual[3] = .48417178779574693
        desired = list(actual)
        desired[3] = .4686765258896835
        trace.observe(NS(joint_names=JOINTS, header=NS(stamp=NS(sec=100, nanosec=0)),
                         actual=NS(positions=actual, velocities=[]),
                         desired=NS(positions=desired, velocities=[]),
                         error=NS(positions=[a-b for a, b in zip(desired, actual)], velocities=[])))
        previous = list(actual)
        previous[3] = .4699574375182084
        guard = StreamGuard(previous, .04, 0, command_velocity_limit=.25)
        guard.command(previous, previous, 100., 100., 0.)
        with self.assertRaisesRegex(AgentError, 'velocity limit'):
            guard.command(actual, actual, 100.009948015, 100.009948015, .01)
        self.assertEqual(guard.last_command, tuple(previous))
        self.assertIn('joint4 error -0.015495 rad', trace.hold_transition_reason(actual, 100.01))
        self.assertIsNone(trace.hold_transition_reason(previous, 100.01))
        self.assertIsNone(trace.hold_transition_reason(actual, 100.06))
        self.assertIsNone(trace.hold_transition_reason(actual, 99.99))
        trace.samples[-1]['error_positions_rad'] = [.014]*7
        self.assertIsNone(trace.hold_transition_reason(actual, 100.01))

    def message(self):
        return NS(joint_names=list(reversed(JOINTS)), header=NS(stamp=NS(sec=100, nanosec=123)),
                  desired=NS(positions=list(range(7)), velocities=[.01]*7),
                  actual=NS(positions=[x-.01 for x in range(7)], velocities=[]),
                  error=NS(positions=[.01]*7, velocities=[]))

    def test_controller_capture_is_ordered_bounded_and_copies_mutable_message(self):
        trace = ControllerTrace(capacity=2)
        msg = self.message()
        for index in range(3):
            trace.observe(msg, index, 100+index)
        snapshot = trace.snapshot()
        self.assertEqual(len(snapshot['samples']), 2)
        row = snapshot['samples'][-1]
        self.assertEqual(row['desired_positions_rad'], list(reversed(range(7))))
        self.assertAlmostEqual(row['controller_stamp_unix_s'], 100.000000123)
        self.assertEqual(row['received_monotonic_s'], 2)
        self.assertIsNone(row['actual_velocities_rad_s'])
        msg.desired.positions[0] = 99
        trace.observe(msg, 4, 104)
        self.assertEqual(row['desired_positions_rad'][-1], 0)
        self.assertEqual(len(snapshot['samples']), 2)

    def test_invalid_message_does_not_interrupt_control(self):
        trace = ControllerTrace()
        trace.observe(NS())
        msg = self.message(); msg.actual.positions[0] = float('nan')
        trace.observe(msg)
        self.assertEqual(trace.invalid_samples, 2)
        self.assertEqual(list(trace.samples), [])

    def test_tracking_rejection_retains_exact_feedback_timestamps(self):
        guard = StreamGuard([0.]*7, .04, 0)
        with self.assertRaisesRegex(AgentError, 'tracking'):
            guard.command([.016]+[0.]*6, [0.]*7, 100.02, 100.03, .03,
                          position_timestamps=[100.01, 100.011, 100.012, 100.013])
        row = guard.history[-1]
        self.assertEqual(row['joint_position_timestamps'][0], 100.01)
        self.assertEqual(row['received_unix_s'], 100.03)
        self.assertEqual(row['position_error_rad'][0], .016)
        self.assertIsNone(guard.last_command)

    def test_command_jump_records_step_and_timing_without_accepting_it(self):
        guard = StreamGuard([0.]*7, .04, 0)
        guard.command([0.]*7, [0.]*7, 100., 100., 0.)
        with self.assertRaisesRegex(AgentError, 'largest step'):
            guard.command([.01]+[0.]*6, [.01]+[0.]*6, 100.01, 100.01, .01)
        row = guard.history[-1]
        self.assertAlmostEqual(row['command_dt_s'], .01)
        self.assertEqual(row['max_command_delta_rad'], .01)
        self.assertEqual(row['previous_command_stamp'], 100.)
        self.assertEqual(guard.last_stamp, 100.)

class TrackingMeanTests(unittest.TestCase):
    def test_peak_is_filtered_and_sustained_error_trips(self):
        from nero_agent.tracking_diagnostic import TrackingMean
        f = TrackingMean()
        def row(t, e):
            return {'controller_stamp_unix_s': t, 'error_positions_rad': [e]*7}
        for i in range(4):
            self.assertIsNone(f.observe(row(i*.02, .014)))
        self.assertIsNone(f.observe(row(.08, .015004)))
        self.assertIsNone(f.observe(row(.08, .1)))  # duplicate does not count
        for i in range(4):
            result = f.observe(row(.1+i*.02, -.016))
        self.assertIn('mean tracking error', result)
        self.assertEqual(len(f.samples), 5)
        self.assertIn('1 samples', f.observe(row(1., .016)))

    def test_no_zero_padding_or_signed_cancellation(self):
        from nero_agent.tracking_diagnostic import TrackingMean
        f = TrackingMean()
        for t, e in ((0., .016), (.02, -.016)):
            self.assertIsNotNone(f.observe({'controller_stamp_unix_s': t,
                                           'error_positions_rad': [e]*7}))
