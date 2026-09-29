import unittest

from nero_agent.settling import SettlingDiagnostics


class SettlingTests(unittest.TestCase):
    def test_position_and_speed_failures_and_dwell_resets(self):
        diagnostic = SettlingDiagnostics([0.]*7, .005, .003, .5)
        def sample(t, position=0., speed=0.):
            diagnostic.observe({'joints_rad': [position]+[0.]*6,
                                'velocities_rad_s': [0., -speed]+[0.]*5}, t)
        sample(0.)
        sample(.2)
        sample(.3, speed=.004)
        sample(.4, position=.006)
        sample(.7, position=.006)
        sample(.8)
        sample(1.)
        result = diagnostic.result()
        self.assertAlmostEqual(result['longest_standstill_s'], .6)
        self.assertAlmostEqual(result['longest_at_goal_and_still_s'], .2)
        self.assertEqual(result['joints']['joint1']['position_failure_samples'], 2)
        self.assertEqual(result['joints']['joint2']['speed_failure_samples'], 1)
        self.assertEqual(result['joints']['joint1']['peak_position_error_rad'], .006)
        self.assertEqual(result['joints']['joint2']['peak_speed_rad_s'], .004)
        self.assertEqual(result['samples'], 7)
        self.assertEqual(result['last_sample']['position_errors_rad'], [0.]*7)

    def test_no_samples_are_explicit(self):
        result = SettlingDiagnostics([0.]*7, .005, .003, .5).result()
        self.assertEqual(result['samples'], 0)
        self.assertIsNone(result['last_sample'])
