#!/usr/bin/env python3
"""Reset Nero's motion-controller state without enabling or moving."""

import time

from nero_safety_common import connect_nero, disconnect_nero


STARTUP_TIMEOUT = 15.0
RESET_TIMEOUT = 5.0


def wait_for_status(robot, timeout):
    """Wait until the CAN reader has received an arm-status frame."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = robot.get_arm_status()
        if status is not None:
            return status
        time.sleep(0.1)
    raise TimeoutError(
        "No arm-status feedback received. Check CAN wiring, bitrate, "
        "adapter ownership, and the Nero power/green status light."
    )


def main():
    robot = None
    try:
        print("Connecting to Nero...")
        robot = connect_nero()
        print("Connected")

        # Do not send reset until the reader has received feedback.  A newly
        # opened USB-CAN adapter can connect successfully while its receive
        # path is not ready yet.
        before = wait_for_status(robot, STARTUP_TIMEOUT)
        print("State before reset:", before.msg.arm_status)

        input(
            "Press Enter to send reset (the arm may lose power/drop), "
            "or Ctrl-C to abort: "
        )
        robot.reset()
        print("Reset command sent; checking arm state...")

        deadline = time.monotonic() + RESET_TIMEOUT
        last_state = None
        while time.monotonic() < deadline:
            status = robot.get_arm_status()
            if status is not None:
                arm_state = status.msg.arm_status
                if arm_state != last_state:
                    print("Current arm state:", arm_state)
                    last_state = arm_state
                if arm_state == robot.ARM_STATUS.ArmStatus.NORMAL:
                    print("Motion-controller state cleared. Motors remain disabled.")
                    return
            time.sleep(0.1)

        if last_state == robot.ARM_STATUS.ArmStatus.NO_SOLUTION:
            raise RuntimeError(
                "Arm remains in NO_SOLUTION(0x2). This is an inverse-kinematics "
                "or unreachable-target state, not an emergency-stop state. "
                "Inspect the last target/current pose and do not resend the "
                "same motion."
            )
        if last_state == robot.ARM_STATUS.ArmStatus.EMERGENCY_STOP:
            raise RuntimeError(
                "Arm remains in EMERGENCY_STOP(0x1). Release the physical "
                "emergency stop and clear the UI emergency stop, then run "
                "this reset again."
            )
        raise RuntimeError(
            f"Arm did not return to NORMAL; last reported state: {last_state}."
        )
    finally:
        if robot is not None:
            disconnect_nero(robot)
            print("Disconnected")


if __name__ == "__main__":
    main()
