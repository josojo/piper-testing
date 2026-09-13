#!/usr/bin/env python3
"""Try to recover Nero from NO_SOLUTION using software only.

This script never enables, disables, or moves the arm.  It waits for CAN
feedback, prints read-only diagnostics, sends one motion-controller reset,
then waits for the resulting arm state.
"""

import time

from nero_safety_common import connect_nero, disconnect_nero


FEEDBACK_TIMEOUT = 15.0
RESET_TIMEOUT = 5.0


def wait_for_status(robot, timeout=FEEDBACK_TIMEOUT):
    """Wait for the CAN reader to receive an arm-status frame."""
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


def print_diagnostics(robot):
    """Print read-only state useful for diagnosing NO_SOLUTION."""
    status = robot.get_arm_status()
    print("Arm status:", status.msg if status is not None else None)

    joints = robot.get_joint_angles()
    print("Joint angles:", joints.msg if joints is not None else None)

    pose = robot.get_flange_pose()
    print("Flange pose:", pose.msg if pose is not None else None)

    driver_states = []
    for joint_index in range(1, robot.joint_nums + 1):
        state = robot.get_driver_states(joint_index)
        driver_states.append(state.msg if state is not None else None)
    print("Driver states:", driver_states)


def main():
    robot = None
    try:
        print("Connecting to Nero...")
        robot = connect_nero()
        print("Connected; waiting for CAN feedback...")

        before = wait_for_status(robot)
        print("State before reset:", before.msg.arm_status)
        print_diagnostics(robot)

        print()
        print("This trial will only send robot.reset().")
        print("It will not enable motors or send a motion command.")
        input("Press Enter to continue, or Ctrl-C to abort: ")

        robot.reset()
        print("Reset command sent; waiting for arm state...")

        deadline = time.monotonic() + RESET_TIMEOUT
        last_state = None
        while time.monotonic() < deadline:
            status = robot.get_arm_status()
            if status is not None:
                last_state = status.msg.arm_status
                print("Current arm state:", last_state)
                if last_state == robot.ARM_STATUS.ArmStatus.NORMAL:
                    print("Recovered to NORMAL. Motors remain disabled.")
                    print_diagnostics(robot)
                    return
            time.sleep(0.1)

        print("Reset did not produce NORMAL.")
        print("Last state:", last_state)
        print_diagnostics(robot)

        if last_state == robot.ARM_STATUS.ArmStatus.NO_SOLUTION:
            print(
                "NO_SOLUTION remains active. Do not enable or move the arm; "
                "inspect the target/current pose and check for another "
                "process sending commands."
            )
        elif last_state == robot.ARM_STATUS.ArmStatus.EMERGENCY_STOP:
            print(
                "The arm is still in emergency stop. Release the physical "
                "and UI emergency stops, then retry."
            )
    finally:
        if robot is not None:
            disconnect_nero(robot)
            print("Disconnected")


if __name__ == "__main__":
    main()
