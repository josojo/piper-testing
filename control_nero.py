#!/usr/bin/env python3
"""Interactive, safety-checked joint control for an AgileX Nero arm.

Joint numbers entered by the user are 1-7.  Angles entered at the prompt are
offsets in degrees relative to the latest measured joint angle.
"""

import math
import os
import platform
import time

from nero_safety_common import connect_nero, disconnect_nero


# Measure this in the Nero base coordinate frame before operating the arm.
PLATFORM_TOP_Z = 0.001       # metres
TOOL_LOWEST_POINT = 0.0      # metres below the flange
SAFETY_MARGIN = 0.030        # metres above the table
SPEED_PERCENT = 10
MOTION_TIMEOUT = 15.0
STARTUP_TIMEOUT = 15.0
RESET_TIMEOUT = 5.0
DISABLE_ON_EXIT = True


def wait_for_arm_status(robot, timeout=STARTUP_TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = robot.get_arm_status()
        if status is not None:
            return status
        time.sleep(0.1)
    raise TimeoutError(
        "No arm-status feedback. Check CAN wiring, bitrate, can0, and the "
        "Nero power/green status light."
    )


def reset_no_solution(robot):
    """Clear a stale controller state before enabling the joints."""
    print("Arm reports NO_SOLUTION; sending controller reset...")
    robot.reset()
    deadline = time.monotonic() + RESET_TIMEOUT
    last_state = None
    while time.monotonic() < deadline:
        status = robot.get_arm_status()
        if status is not None:
            last_state = status.msg.arm_status
            if last_state in (
                robot.ARM_STATUS.ArmStatus.NORMAL,
                robot.ARM_STATUS.ArmStatus.JOINT_BRAKE_NOT_RELEASED,
            ):
                return status
            if last_state == robot.ARM_STATUS.ArmStatus.EMERGENCY_STOP:
                raise RuntimeError(
                    "The arm is in EMERGENCY_STOP. Release the physical/UI "
                    "emergency stop and reset the arm before running this script."
                )
        time.sleep(0.1)
    raise TimeoutError(
        f"Reset did not clear NO_SOLUTION (last state: {last_state}). "
        "Check the Nero firmware version configured in nero_safety_common.py."
    )


def wait_for_normal_state(robot, timeout=STARTUP_TIMEOUT):
    deadline = time.monotonic() + timeout
    last_state = None
    while time.monotonic() < deadline:
        status = robot.get_arm_status()
        if status is not None:
            last_state = status.msg.arm_status
            if last_state == robot.ARM_STATUS.ArmStatus.NORMAL:
                return
            if last_state != robot.ARM_STATUS.ArmStatus.JOINT_BRAKE_NOT_RELEASED:
                raise RuntimeError(f"Arm is not ready: {last_state}")
        time.sleep(0.1)
    raise TimeoutError(f"Arm did not become NORMAL; last state: {last_state}")


def flange_pose_from_joints(robot, joint_angles):
    """Return [x, y, z, roll, pitch, yaw] calculated by SDK forward kinematics."""
    pose = robot.fk(joint_angles)
    if pose is None:
        raise RuntimeError("SDK forward kinematics returned no pose")
    return pose.msg if hasattr(pose, "msg") else pose


def check_table_clearance(robot, joint_angles):
    if PLATFORM_TOP_Z is None:
        raise RuntimeError("Set PLATFORM_TOP_Z before moving the arm")

    pose = flange_pose_from_joints(robot, joint_angles)
    minimum_z = PLATFORM_TOP_Z + SAFETY_MARGIN + TOOL_LOWEST_POINT
    print(f"Predicted flange z: {pose[2]:.4f} m; minimum: {minimum_z:.4f} m")
    if pose[2] < minimum_z:
        raise RuntimeError(
            f"Move rejected: predicted flange z={pose[2]:.4f} m is below "
            f"the table safety height {minimum_z:.4f} m"
        )
    return pose


def wait_motion_done(robot, timeout=MOTION_TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = robot.get_arm_status()
        if status is not None:
            if status.msg.arm_status != robot.ARM_STATUS.ArmStatus.NORMAL:
                raise RuntimeError(f"Arm left NORMAL state: {status.msg.arm_status}")
            if getattr(status.msg, "err_code", 0):
                raise RuntimeError(f"Arm reported an error: {status.msg}")
            if getattr(status.msg, "motion_status", None) == 0:
                return
        time.sleep(0.05)
    raise TimeoutError(f"Motion did not complete within {timeout:.1f} seconds")


def print_angles(robot):
    joints = robot.get_joint_angles()
    if joints is None:
        raise RuntimeError("No joint-angle feedback received")
    angles = list(joints.msg)
    degrees = [math.degrees(angle) for angle in angles]
    print("\nCurrent joint angles:")
    for index, (radians, degree) in enumerate(zip(angles, degrees), start=1):
        print(f"  Joint {index}: {radians:+.5f} rad ({degree:+.2f} deg)")
    return angles


def validate_before_move(robot):
    status = robot.get_arm_status()
    if status is None or status.msg.arm_status != robot.ARM_STATUS.ArmStatus.NORMAL:
        state = status.msg.arm_status if status is not None else None
        raise RuntimeError(f"Arm is not NORMAL before move: {state}")
    if not robot.is_ok():
        raise RuntimeError("Arm is not OK; move rejected")
    if not all(robot.get_joints_enable_status_list()):
        raise RuntimeError("Not all joints are enabled; move rejected")


def main():
    robot = None
    connected = False
    try:
        print(f"Configuration: {platform.system()} CAN backend")
        print(
            "  interface="
            f"{os.environ.get('NERO_CAN_INTERFACE', 'socketcan')}, "
            f"channel={os.environ.get('NERO_CAN_CHANNEL', 'can0')}"
        )
        print("Connecting...")
        robot = connect_nero()
        connected = True
        print("Connected")

        status = wait_for_arm_status(robot)
        if status.msg.arm_status == robot.ARM_STATUS.ArmStatus.NO_SOLUTION:
            status = reset_no_solution(robot)
        elif status.msg.arm_status == robot.ARM_STATUS.ArmStatus.EMERGENCY_STOP:
            raise RuntimeError(
                "The arm is in EMERGENCY_STOP. Release the physical/UI emergency "
                "stop and reset the arm before running this script."
            )
        elif status.msg.arm_status not in (
            robot.ARM_STATUS.ArmStatus.NORMAL,
            robot.ARM_STATUS.ArmStatus.JOINT_BRAKE_NOT_RELEASED,
        ):
            raise RuntimeError(f"Arm starts in an unsafe state: {status.msg.arm_status}")

        print("Enabling all joints...")
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while not all(robot.get_joints_enable_status_list()):
            if time.monotonic() >= deadline:
                raise TimeoutError("Not all joints became enabled")
            robot.enable()
            time.sleep(0.05)
        wait_for_normal_state(robot)
        robot.set_speed_percent(SPEED_PERCENT)
        print("All joints enabled. Type q to quit.")

        while True:
            current = print_angles(robot)
            try:
                joint_text = input("Choose a joint (1-7), or q to quit: ").strip().lower()
            except EOFError:
                break
            if joint_text in ("q", "quit", "exit"):
                break
            try:
                joint_number = int(joint_text)
                if not 1 <= joint_number <= 7:
                    raise ValueError
                delta_degrees = float(input("Move this joint by how many degrees? "))
                if not math.isfinite(delta_degrees) or abs(delta_degrees) > 20:
                    raise ValueError("enter a finite offset between -20 and 20 degrees")
            except ValueError as exc:
                print(f"Invalid input: {exc or 'joint must be 1-7'}")
                continue

            target = current.copy()
            target[joint_number - 1] += math.radians(delta_degrees)
            print(f"Requested target: joint {joint_number} by {delta_degrees:+.2f} degrees")
            try:
                pose = check_table_clearance(robot, target)
                print("Predicted target flange pose [m, rad]:", pose)
                validate_before_move(robot)
            except (RuntimeError, ValueError) as exc:
                print(exc)
                continue

            confirmation = input("Press Enter to move, or type n to cancel: ").strip().lower()
            if confirmation in ("n", "no", "cancel"):
                print("Move cancelled")
                continue

            # Refresh and recheck after the user confirmation because the arm
            # may have changed while the prompt was open.
            current = print_angles(robot)
            target = current.copy()
            target[joint_number - 1] += math.radians(delta_degrees)
            try:
                check_table_clearance(robot, target)
                validate_before_move(robot)
                robot.move_j(target)
                wait_motion_done(robot)
                print("Motion completed")
            except (RuntimeError, TimeoutError, ValueError) as exc:
                print(f"Move failed or was rejected: {exc}")
    finally:
        if connected and robot is not None:
            if DISABLE_ON_EXIT:
                try:
                    print("Disabling all joints before disconnect...")
                    robot.disable()
                except Exception as exc:
                    print(f"Warning: could not disable joints cleanly: {exc}")
            disconnect_nero(robot)
            print("Disconnected")


if __name__ == "__main__":
    main()
