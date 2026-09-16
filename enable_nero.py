"""Enable all NERO joints without commanding a motion.

The motors remain enabled after disconnect so the experiment can perform its
own fresh state check. Keep the arm supported and supervised while running.
"""

import time

from nero_safety_common import connect_nero, disconnect_nero


ENABLE_TIMEOUT_S = 10.0
NORMAL_TIMEOUT_S = 5.0


def main() -> int:
    robot = None
    try:
        print("Connecting to NERO...")
        robot = connect_nero()
        print("Joint states before enable:", robot.get_joints_enable_status_list())
        if not all(robot.get_joints_enable_status_list()):
            print("Enabling all motor joints...")
            deadline = time.monotonic() + ENABLE_TIMEOUT_S
            while time.monotonic() < deadline:
                if robot.enable() and all(robot.get_joints_enable_status_list()):
                    break
                time.sleep(0.05)
        joints = robot.get_joints_enable_status_list()
        if not all(joints):
            raise RuntimeError(f"Could not enable all joints: {joints}")

        deadline = time.monotonic() + NORMAL_TIMEOUT_S
        last_state = None
        while time.monotonic() < deadline:
            status = robot.get_arm_status()
            if status is not None:
                last_state = status.msg.arm_status
                if last_state == robot.ARM_STATUS.ArmStatus.NORMAL:
                    print("All seven joints enabled; arm status is NORMAL.")
                    print("Disconnecting without moving or disabling the joints.")
                    return 0
                if last_state != robot.ARM_STATUS.ArmStatus.JOINT_BRAKE_NOT_RELEASED:
                    raise RuntimeError(f"Arm is not ready after enable: {last_state}")
            time.sleep(0.1)
        raise TimeoutError(f"Arm did not reach NORMAL (last state: {last_state})")
    finally:
        if robot is not None:
            disconnect_nero(robot)


if __name__ == "__main__":
    raise SystemExit(main())
