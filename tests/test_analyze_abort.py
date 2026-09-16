import unittest
from nero_agent.analyze_abort import analyze


class AnalyzeAbortTests(unittest.TestCase):
    def test_signed_peak_and_missing_or_separate_timestamps(self):
        sample = {'monotonic_s': 2., 'joints_rad': [.01]*7,
                  'velocities_rad_s': [-.04]*7}
        report = {'controlled_abort': {
            'hold_target_rad': [0.]*7,
            'timings': {'move_j_returned_monotonic_s': 1.},
            'commissioning': {'history': [sample]}}}
        result = analyze(report)
        self.assertFalse(result['deliberate_trigger_observed'])
        self.assertEqual(result['joints'][0]['peak_reported_velocity_rad_s'], -.04)
        self.assertIsNone(result['joints'][0]['motor_minus_position_timestamp_s'])
        sample.update(joint_position_timestamps=[1.]*4, motor_velocity_timestamps=[1.02]*7)
        self.assertAlmostEqual(analyze(report)['joints'][6]['motor_minus_position_timestamp_s'], .02)

    def test_missing_history_rejected(self):
        with self.assertRaises(ValueError):
            analyze({'controlled_abort': {}})
