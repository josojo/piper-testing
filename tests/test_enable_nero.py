"""Gripper enable checks without hardware connections."""
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

import enable_nero


def feedback(timestamp, enabled=False, mode="width", value=0.032):
    return NS(timestamp=timestamp, msg=NS(
        value=value, mode=mode,
        foc_status=NS(driver_enable_status=enabled)))


class GripperEnableTests(unittest.TestCase):
    def test_enables_at_measured_position_in_each_mode(self):
        for mode, method, value in [("width", "move_gripper_m", 0.032),
                                    ("angle", "move_gripper_deg", 12.5)]:
            with self.subTest(mode=mode):
                gripper = MagicMock()
                gripper.get_gripper_status.side_effect = [
                    feedback(101, mode=mode, value=value),
                    feedback(103, enabled=True, mode=mode, value=value)]
                with patch.object(enable_nero.time, "time", side_effect=[100, 102]), \
                        patch.object(enable_nero.time, "sleep"):
                    enable_nero.enable_gripper(gripper)
                getattr(gripper, method).assert_called_once_with(value=value, force=1.0)

    def test_missing_or_stale_feedback_never_commands_a_target(self):
        for packet in [None, feedback(99, enabled=True)]:
            with self.subTest(packet=packet):
                gripper = MagicMock()
                gripper.get_gripper_status.return_value = packet
                with patch.object(enable_nero.time, "time", return_value=100), \
                        patch.object(enable_nero.time, "monotonic", side_effect=[0, 1, 11]), \
                        patch.object(enable_nero.time, "sleep"):
                    with self.assertRaises(TimeoutError):
                        enable_nero.enable_gripper(gripper)
                gripper.move_gripper_m.assert_not_called()
                gripper.move_gripper_deg.assert_not_called()

    def test_fresh_enabled_feedback_needs_no_command(self):
        gripper = MagicMock()
        gripper.get_gripper_status.return_value = feedback(101, enabled=True)
        with patch.object(enable_nero.time, "time", return_value=100):
            enable_nero.enable_gripper(gripper)
        gripper.move_gripper_m.assert_not_called()
        gripper.move_gripper_deg.assert_not_called()
