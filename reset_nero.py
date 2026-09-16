#!/usr/bin/env python3
"""Reset Nero's motion controller and attached gripper without position commands."""

from copy import deepcopy
import time

STARTUP_TIMEOUT = 15.0
RESET_TIMEOUT = 5.0


def wait_for_gripper(gripper, timeout, after=None):
    """Require fresh feedback, and after reset require a disabled driver."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = deepcopy(gripper.get_gripper_status())
        if status is not None and 0 <= time.time() - status.timestamp <= 0.25:
            if after is None or (status.timestamp > after and
                                 not status.msg.foc_status.driver_enable_status):
                return status
        time.sleep(0.1)
    raise TimeoutError('No fresh gripper feedback' if after is None else
                       'Gripper reset could not be verified: no fresh disabled-driver feedback')


def reset_gripper(gripper):
    wait_for_gripper(gripper, STARTUP_TIMEOUT)
    sent_at = time.time()
    # The SDK boolean reflects cached pre-command feedback, not an acknowledgement.
    gripper.reset_gripper()
    wait_for_gripper(gripper, RESET_TIMEOUT, after=sent_at)
    print('Gripper reset sent; fresh feedback confirms its driver is disabled.')


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
    from nero_safety_common import connect_nero, disconnect_nero
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
        gripper = robot.init_effector(robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
        wait_for_gripper(gripper, STARTUP_TIMEOUT)

        input(
            "Press Enter to reset the arm and gripper (the arm may lose power/drop "
            "and the gripper may release its load), "
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
                    print("Motion-controller state cleared; no enable command sent.")
                    reset_gripper(gripper)
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
