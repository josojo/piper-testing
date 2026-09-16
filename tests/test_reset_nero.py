"""Reset verification tests without SDK connections or hardware writes."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

import reset_nero


class GripperResetTests(unittest.TestCase):
    def test_cached_sdk_success_requires_new_feedback(self):
        gripper = MagicMock()
        gripper.reset_gripper.return_value = True
        with patch.object(reset_nero, 'wait_for_gripper', side_effect=[object(), TimeoutError('stale')]) as wait, \
                patch.object(reset_nero.time, 'time', return_value=100):
            with self.assertRaisesRegex(TimeoutError, 'stale'):
                reset_nero.reset_gripper(gripper)
        gripper.reset_gripper.assert_called_once_with()
        self.assertEqual(wait.call_args.kwargs, {'after': 100})

    def test_old_disabled_feedback_is_not_reset_confirmation(self):
        packet = NS(timestamp=100., msg=NS(foc_status=NS(driver_enable_status=False)))
        gripper = NS(get_gripper_status=lambda: packet)
        with patch.object(reset_nero.time, 'time', return_value=100.1), \
                patch.object(reset_nero.time, 'monotonic', side_effect=[0, 0.1, 6]), \
                patch.object(reset_nero.time, 'sleep'):
            with self.assertRaises(TimeoutError):
                reset_nero.wait_for_gripper(gripper, 5, after=100.)

    def test_new_disabled_feedback_confirms_reset(self):
        packet = NS(timestamp=100.1, msg=NS(foc_status=NS(driver_enable_status=False)))
        gripper = NS(get_gripper_status=lambda: packet)
        with patch.object(reset_nero.time, 'time', return_value=100.2):
            self.assertEqual(reset_nero.wait_for_gripper(gripper, 5, after=100.).timestamp, 100.1)
