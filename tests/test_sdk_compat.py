"""Offline checks of the actual SDK encoder/decoder; never open CAN."""

import importlib.util
from types import SimpleNamespace as NS
import unittest

from nero_planner import PlanningError
from nero_experiment.sdk_compat import acceleration_counts, write_joint_acceleration


class AccelerationTests(unittest.TestCase):
    def test_invalid_acceleration_cannot_be_encoded(self):
        for value in (None, True, -0.1, 0, 0.009, float("nan"), float("inf"), 327.67):
            with self.subTest(value=value), self.assertRaises(PlanningError):
                acceleration_counts(value)
        self.assertEqual(acceleration_counts(0.1509), 15)

    @unittest.skipUnless(importlib.util.find_spec("pyAgxArm"), "hardware SDK not installed")
    def test_wire_bytes_and_real_nero_decoder_agree(self):
        from pyAgxArm.protocols.can_protocol.drivers.nero.default.parser import Codec
        from pyAgxArm.protocols.can_protocol.msgs.nero.default import (
            ArmMsgJointConfig, ArmMsgFeedbackAllCurrentMotorMaxAccLimit,
        )
        codec = Codec()
        frames = []

        def send(msg):
            frames.append(codec.encode_475_joint_config(msg))

        def read(joint, **kwargs):
            self.assertEqual(kwargs["min_interval"], 0)
            # Firmware converts centiradian write units to milliradian feedback units.
            raw_value = int.from_bytes(bytes(frames[-1][3:5]), "big") * 10
            raw = list(raw_value.to_bytes(2, "big"))
            decoded = ArmMsgFeedbackAllCurrentMotorMaxAccLimit()
            codec.decode_47C_motor_max_acc_limit(decoded, bytearray([joint] + raw + [0] * 5))
            return NS(timestamp=100, msg=decoded.joints[joint - 1])

        robot = NS(_MSG_JointConfig=ArmMsgJointConfig, _send_msg=send, get_joint_acc_limits=read)
        self.assertEqual(write_joint_acceleration(robot, 4, 0.15, lambda: 100), 0.15)
        # J4, no zero calibration, enable acceleration, raw 15, no clear-error.
        self.assertEqual(frames, [[4, 0, 0xAE, 0, 15, 0, 0, 0]])

    def test_tenfold_readback_is_rejected(self):
        robot = NS(_MSG_JointConfig=NS, _send_msg=lambda msg: None,
                   get_joint_acc_limits=lambda *a, **k: NS(
                       timestamp=100, msg=NS(joint_index=4, max_joint_acc=1.5)))
        with self.assertRaisesRegex(PlanningError, "readback mismatch"):
            write_joint_acceleration(robot, 4, 0.15, lambda: 100)

    def test_delayed_configuration_is_verified_without_resending(self):
        now, frames = [100.0], []
        values = iter((2.5, 2.5, 0.15))

        def read(*args, **kwargs):
            return NS(timestamp=now[0], msg=NS(joint_index=4, max_joint_acc=next(values)))

        def sleep(seconds):
            now[0] += seconds

        robot = NS(_MSG_JointConfig=NS, _send_msg=frames.append, get_joint_acc_limits=read)
        actual = write_joint_acceleration(robot, 4, 0.15, lambda: now[0], lambda: now[0], sleep)
        self.assertEqual(actual, 0.15)
        self.assertEqual(len(frames), 1)
        self.assertGreater(now[0], 100)
