"""NERO 1.21 acceleration compatibility for the reviewed pyAgxArm revision.

NERO 1.21 uses 0.01 rad/s² write units (0x475) and 0.001 rad/s²
feedback units (0x47C). The SDK setter incorrectly multiplies by 1e4
instead of 1e2. Confirmed by hardware readback: write 150 -> read 1.5.
Do not use the upstream setter or modify installed site-packages.
"""

from copy import deepcopy
from decimal import Decimal, ROUND_FLOOR
from importlib.metadata import distribution
import json
import math
import time

from nero_planner import PlanningError


SDK_COMMIT = "e7aef17d54cac80cbaeb1b4110ab3d8f1337a95b"
ACCELERATION_COUNTS_PER_RAD_S2 = 100


def check_sdk_revision():
    package = distribution("pyAgxArm")
    origin = json.loads(package.read_text("direct_url.json") or "{}")
    if package.version != "1.0.0" or origin.get("vcs_info", {}).get("commit_id") != SDK_COMMIT:
        raise PlanningError("Install the reviewed pyAgxArm revision from requirements-hardware.txt "
                            "before using the NERO hardware adapter")


def acceleration_counts(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PlanningError("Joint acceleration must be a finite positive number")
    # Round downward so quantization can never increase a requested cap.
    raw = int((Decimal(str(value)) * ACCELERATION_COUNTS_PER_RAD_S2).to_integral_value(rounding=ROUND_FLOOR))
    if not 1 <= raw < 0x7FFF:
        raise PlanningError("Joint acceleration must be in [0.01, 327.67) rad/s²")
    return raw


def read_joint_acceleration(robot, joint, wall_clock=time.time):
    requested_at = wall_clock()
    packet = deepcopy(robot.get_joint_acc_limits(joint, timeout=1.0, min_interval=0.0))
    if packet is None:
        raise PlanningError(f"Missing acceleration readback for joint {joint}")
    if (not math.isfinite(packet.timestamp) or packet.timestamp < requested_at
            or not 0 <= wall_clock() - packet.timestamp <= 0.25):
        raise PlanningError(f"Stale acceleration readback for joint {joint}")
    if packet.msg.joint_index != joint:
        raise PlanningError(f"Wrong joint in acceleration readback for joint {joint}")
    value = packet.msg.max_joint_acc
    acceleration_counts(value)  # Reject missing, zero, nonfinite and sentinel values.
    return float(value)


def write_joint_acceleration(robot, joint, limit, wall_clock=time.time,
                             clock=time.monotonic, sleep=time.sleep):
    if type(joint) is not int or not 1 <= joint <= 7:
        raise PlanningError("Acceleration writes require an individual joint from 1 to 7")
    raw = acceleration_counts(limit)
    # Only the acceleration field is enabled. Never calibrate zero or clear faults.
    robot._send_msg(robot._MSG_JointConfig(
        joint_index=joint, set_motor_current_pos_as_zero=0,
        acc_param_config_is_effective_or_not=0xAE,
        max_joint_acc=raw, clear_joint_err=0,
    ))
    expected = raw / ACCELERATION_COUNTS_PER_RAD_S2
    deadline = clock() + 1.0
    while True:
        actual = read_joint_acceleration(robot, joint, wall_clock)
        if math.isclose(actual, expected, abs_tol=1e-9, rel_tol=0):
            return actual
        if clock() >= deadline:
            raise PlanningError(f"Joint {joint} acceleration readback mismatch: requested "
                                f"{expected:.3f}, received {actual:.3f} rad/s²; motion prohibited")
        # Configuration is asynchronous. Re-query; never resend or increase a cap.
        sleep(0.02)
