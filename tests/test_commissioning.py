import unittest
from types import SimpleNamespace as NS
from nero_agent.commissioning import MovingAbortTrial, validate_trial_plan
from nero_agent.core import AgentError, JOINTS


def state(t, q=.001, v=.01):
    return NS(joints_rad=[q]+[0.]*6, velocities_rad_s=[v]+[0.]*6,
              enabled=[True]*7, feedback_timestamps=[t]*4)


class MovingAbortTests(unittest.TestCase):
    def test_trace_preserves_separate_packet_times_and_command_context(self):
        trial = MovingAbortTrial([0.]*7, 0)
        trial.command([0.]*7, .01, .02)
        sample = state(.1)
        sample.feedback_timestamps = [.1, .101, .102, .103]
        sample.motion_status = 1
        sample.joint_position_timestamps = [.10]*4
        sample.motor_velocity_timestamps = [.09]*7
        trial.observe(sample, .12)
        saved = trial.history[-1]
        self.assertEqual(saved['motor_velocity_timestamps'], [.09]*7)
        self.assertEqual(saved['joint_position_timestamps'], [.10]*4)
        self.assertEqual(saved['last_validated_stream_command']['ros_header_stamp_s'], .01)
        self.assertEqual(saved['last_validated_stream_command']['monotonic_s'], .02)
        self.assertEqual(saved['motion_status'], 1)
        self.assertEqual(len(saved['feedback_packet_age_s']), 4)
        self.assertAlmostEqual(saved['feedback_packet_skew_s'], .003)
        self.assertNotIn('history', trial.live_status())

    def trigger(self):
        trial = MovingAbortTrial([0.]*7, 0.)
        self.assertIsNone(trial.observe(state(.1), .1))
        self.assertIsNotNone(trial.observe(state(.2, .002), .2))
        self.assertEqual(trial.outcome, 'triggered')
        return trial

    def verdict(self, trial, **changes):
        abort = dict(status='holding', stable_since_monotonic_s=.4,
                     controller_cancellation='acknowledged')
        abort.update(changes)
        return trial.result(abort)['status']

    def test_trigger_then_sustained_hold(self):
        trial = self.trigger()
        trial.observe(state(.4, .003, 0), .4)
        self.assertEqual(self.verdict(trial), 'passed')
        self.assertFalse(trial.result({'status':'settling'})['normal_execution_unlocked'])

    def test_same_feedback_cannot_trigger(self):
        trial = MovingAbortTrial([0.]*7, 0.)
        trial.observe(state(.1), .1)
        trial.observe(state(.1), .2)
        self.assertEqual(trial.outcome, 'pending')
        trial.observe(state(.1), 4.1)
        self.assertEqual(trial.outcome, 'inconclusive')

    def test_no_motion_and_goal_arrival_are_inconclusive(self):
        for sample, t in ((state(4.1, 0, 0),4.1), (state(.1,.009,.01),.1)):
            trial = MovingAbortTrial([0.]*7, 0.)
            trial.observe(sample,t)
            self.assertEqual(self.verdict(trial), 'inconclusive')

    def test_bounds_trip_is_not_deliberate_pass(self):
        for q,v in ((.001,.04),(.02,.01)):
            trial = MovingAbortTrial([0.]*7,0)
            trial.observe(state(.1,q,v),.1)
            self.assertEqual(self.verdict(trial),'failed')

    def test_unconfirmed_cancel_slow_stop_and_lost_hold_fail(self):
        trial = self.trigger()
        for changes in (dict(controller_cancellation='timeout'),
                        dict(stable_since_monotonic_s=1.3), dict(status='failed')):
            self.assertEqual(self.verdict(trial,**changes),'failed')
        self.assertEqual(trial.result({'status':'holding'},fault='lost feedback')['status'],'failed')

    def test_post_abort_travel_and_speed_fail(self):
        for q,v in ((.02,0),(.003,.04)):
            trial = self.trigger()
            trial.observe(state(.4,q,v),.4)
            self.assertEqual(self.verdict(trial),'failed')

    def test_feedback_failure_does_not_hide_motion_violations(self):
        trial = self.trigger()
        sample = state(.4, .003, 0)
        sample.joints_rad[1] = .00328
        sample.velocities_rad_s[3] = -.055
        trial.observe(sample, .4)
        result = trial.result({'status': 'failed', 'failure': 'stale feedback'})
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['observed_violations']), 2)
        self.assertFalse(result['metrics']['standstill_dwell_confirmed'])

    def test_commands_bounded_before_sdk_write(self):
        trial = MovingAbortTrial([0.]*7,0)
        trial.command([0.]*7,1.)
        trial.command([.0001]+[0.]*6,1.01)
        with self.assertRaises(AgentError):
            trial.command([.01]+[0.]*6,1.02)
        with self.assertRaises(AgentError):
            trial.command([0,.003]+[0.]*5,2.)

    def test_plan_cannot_change_other_joints(self):
        def point(q,t):
            return NS(positions=q,velocities=[0.]*7,accelerations=[0.]*7,
                      time_from_start=NS(sec=t,nanosec=0))
        path = NS(joint_names=JOINTS,points=[point([0.]*7,0),point([.01]+[0.]*6,1)])
        validate_trial_plan(path,[0.]*7)
        path.points.insert(1,point([.005,.003]+[0.]*5,0))
        path.points[1].time_from_start.nanosec=500000000
        with self.assertRaisesRegex(AgentError,'fixed joint1'):
            validate_trial_plan(path,[0.]*7)

class CommissioningBackendTests(unittest.TestCase):
    def test_completion_without_trigger_is_inconclusive_and_stops(self):
        from unittest.mock import MagicMock, patch
        from nero_agent.ros_backend import RosBackend
        backend = RosBackend.__new__(RosBackend)
        backend.hardware = True
        backend.settings = NS(namespace='/nero')
        initial = dict(joints_rad=[0.]*7, velocities_rad_s=[0.]*7, gripper_width_m=.02)
        backend.plans = {'trial': (NS(joint_trajectory=object()), initial, [.01]+[0.]*6, 'scene')}
        backend.state = lambda: initial.copy()
        backend._scene_digest = lambda: 'scene'
        backend.node = MagicMock()
        backend.executor = MagicMock()
        backend._call = MagicMock(side_effect=[NS(success=True), NS(message='{"status":"idle"}')])
        handle = MagicMock(accepted=True)
        handle.get_result_async.return_value.done.return_value = True
        backend._wait = lambda *a, **kw: handle
        backend.abort_status = object()
        backend.stop = MagicMock(return_value={'controlled_abort': {'commissioning': {'status':'pending'}}})
        modules = {'std_srvs.srv':NS(Trigger=NS(Request=NS)),
                   'moveit_msgs.action':NS(ExecuteTrajectory=NS(Goal=NS))}
        with patch.dict('sys.modules',modules), patch('nero_agent.commissioning.validate_trial_plan'):
            result = backend.commission_abort({'token':'trial'})
        self.assertEqual(result['status'],'inconclusive')
        backend.stop.assert_called_once()
        self.assertFalse(backend.plans)


if __name__ == '__main__':
    unittest.main()
