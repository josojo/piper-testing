import unittest
from types import SimpleNamespace as NS
from unittest.mock import MagicMock
from nero_agent.feedback_diagnostic import Recorder, PACKETS


class DiagnosticTests(unittest.TestCase):
    def test_delayed_read_and_stale_feedback_preserved_without_writes(self):
        now = [10.]
        parser = NS(**{name:NS(timestamp=100.) for name in PACKETS})
        hardware = NS(robot=NS(_parser=parser))
        def read(require_enabled):
            self.assertFalse(require_enabled)
            now[0] += .06
            raise RuntimeError('stale feedback')
        hardware.read = read
        recorder = Recorder(hardware, lambda:now[0], lambda:100. + now[0]-10.)
        recorder.tick()
        now[0] += .07
        recorder.tick()
        result = recorder.result()
        self.assertEqual(result['summary']['read_errors'],2)
        self.assertAlmostEqual(result['summary']['read_duration_s']['max_s'],.06)
        self.assertAlmostEqual(result['summary']['timer_interval_s']['max_s'],.13)
        self.assertEqual(result['samples'][0]['error'],'stale feedback')
        self.assertFalse(result['motion_commands_sent'])

    def test_missing_packets_and_successful_state(self):
        hardware = NS(robot=NS(), read=MagicMock(return_value=NS(joints_rad=[0.]*7,
                      velocities_rad_s=[0.]*7, enabled=[False]*7)))
        recorder = Recorder(hardware)
        recorder.tick()
        result = recorder.result()
        self.assertEqual(result['summary']['successful_reads'],1)
        self.assertIsNone(result['summary']['packets']['joint_12']['age']['max_s'])
        hardware.read.assert_called_once_with(require_enabled=False)
