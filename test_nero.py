import time
import os
import platform

from nero_safety_common import connect_nero, disconnect_nero

# Safety settings for the first joint-space motion.
#
# Measure PLATFORM_TOP_Z in the Nero base coordinate frame before running.
# Leave it as None until you have measured it; the script will refuse to move.
PLATFORM_TOP_Z = 0.001      # metres; measured platform height
TOOL_LOWEST_POINT = 0.0    # metres below the flange, including any tool
SAFETY_MARGIN = 0.030      # 3 cm above the platform
MOVE_JOINT_INDEX = 0       # zero-based index; 0 is joint 1
MOVE_JOINT_DELTA_RAD = 0.05  # about 2.86 degrees; visibly larger test move
SPEED_PERCENT = 10
ENABLE_MOTORS = True       # set False when enabling manually via the UI
# Disable before closing CAN so Nero is not left in CAN-control mode between
# invocations. Set False only if the arm is mechanically supported and must
# remain enabled after this script exits.
DISABLE_ON_EXIT = True
ENABLE_TIMEOUT = 10.0
ENABLE_RECONNECT_ATTEMPTS = 2
MOTION_TIMEOUT = 10.0
STARTUP_TIMEOUT = 15.0
NORMAL_TIMEOUT = 5.0
RESET_TIMEOUT = 5.0

can_interface = os.environ.get(
      "NERO_CAN_INTERFACE",
      "socketcan" if platform.system() == "Linux" else "slcan",
)
can_channel = os.environ.get(
      "NERO_CAN_CHANNEL",
      "can0" if platform.system() == "Linux" else "/dev/ttyACM0",
)
print(f"Configuration: {platform.system()} CAN backend")
print(f"  interface={can_interface}, channel={can_channel}, bitrate=1000000")
if platform.system() == "Linux":
      print(
            "Ubuntu prerequisite: activate the CAN interface first, for example: "
            "sudo ip link set can0 up type can bitrate 1000000"
      )
robot = None
connected = False


def wait_motion_done(robot, timeout=MOTION_TIMEOUT):
      """Wait for the arm to report that the commanded motion is complete."""
      deadline = time.monotonic() + timeout
      while time.monotonic() < deadline:
            status = robot.get_arm_status()
            if status is not None:
                  if status.msg.arm_status != robot.ARM_STATUS.ArmStatus.NORMAL:
                        raise RuntimeError(
                              f"Arm is not in NORMAL state: {status.msg.arm_status}"
                        )
                  # ``err_status`` is a container object, so it is truthy even
                  # when every individual flag inside it is False.  The
                  # protocol's packed error value is the reliable check.
                  if getattr(status.msg, "err_code", 0):
                        raise RuntimeError(f"Arm reported an error: {status.msg}")
                  if getattr(status.msg, "motion_status", None) == 0:
                        return
            time.sleep(0.05)
      raise TimeoutError(f"Motion did not complete within {timeout:.1f} seconds")


def wait_for_arm_status(robot, timeout=STARTUP_TIMEOUT):
      """Wait for the CAN reader to receive a current arm-status frame."""
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


def wait_for_normal_state(robot, timeout=NORMAL_TIMEOUT):
      """Wait for the controller to become ready after motor enable."""
      deadline = time.monotonic() + timeout
      last_status = None
      while time.monotonic() < deadline:
            status = robot.get_arm_status()
            if status is not None:
                  last_status = status
                  arm_state = status.msg.arm_status
                  if arm_state == robot.ARM_STATUS.ArmStatus.NORMAL:
                        return status
                  if arm_state != robot.ARM_STATUS.ArmStatus.JOINT_BRAKE_NOT_RELEASED:
                        raise RuntimeError(
                              f"Arm is not ready after enable ({arm_state}); "
                              "clear/reset the reported arm state before moving"
                        )
            time.sleep(0.1)
      state = last_status.msg.arm_status if last_status is not None else None
      raise TimeoutError(
            f"Arm did not reach NORMAL after enabling within {timeout:.1f} "
            f"seconds (last state: {state})"
      )


def reset_no_solution(robot, timeout=RESET_TIMEOUT):
      """Clear a stale NO_SOLUTION state before enabling the motors."""
      print("Arm reports NO_SOLUTION; sending one controller reset...")
      robot.reset()
      deadline = time.monotonic() + timeout
      last_state = None
      while time.monotonic() < deadline:
            status = robot.get_arm_status()
            if status is not None:
                  last_state = status.msg.arm_status
                  if last_state in (
                        robot.ARM_STATUS.ArmStatus.NORMAL,
                        robot.ARM_STATUS.ArmStatus.JOINT_BRAKE_NOT_RELEASED,
                  ):
                        print("Reset cleared the stale arm state:", last_state)
                        return status
                  if last_state == robot.ARM_STATUS.ArmStatus.EMERGENCY_STOP:
                        raise RuntimeError(
                              "Reset did not clear EMERGENCY_STOP; release the "
                              "physical/UI emergency stop before continuing"
                        )
            time.sleep(0.1)
      raise TimeoutError(
            f"Reset did not clear NO_SOLUTION within {timeout:.1f} seconds "
            f"(last state: {last_state}). Check the firmware version and "
            "the last commanded target."
      )

try:
      print("Connecting...")
      robot = connect_nero()
      connected = True
      print("Connected")
      startup_status = wait_for_arm_status(robot)
      print("Arm status before enable:", startup_status.msg.arm_status)
      if startup_status.msg.arm_status != robot.ARM_STATUS.ArmStatus.NORMAL:
            startup_state = startup_status.msg.arm_status
            if (
                  startup_state
                  == robot.ARM_STATUS.ArmStatus.JOINT_BRAKE_NOT_RELEASED
                  and ENABLE_MOTORS
            ):
                  print(
                        "Joints are disabled; JOINT_BRAKE_NOT_RELEASED is "
                        "expected until the enable command releases the brakes"
                  )
            elif startup_state == robot.ARM_STATUS.ArmStatus.NO_SOLUTION:
                  startup_status = reset_no_solution(robot)
            elif startup_state == robot.ARM_STATUS.ArmStatus.EMERGENCY_STOP:
                  raise RuntimeError(
                        "Arm starts in EMERGENCY_STOP(0x1); release the "
                        "physical/UI emergency stop and reset before enabling"
                  )
            elif startup_state != robot.ARM_STATUS.ArmStatus.JOINT_BRAKE_NOT_RELEASED:
                  raise RuntimeError(
                        f"Arm starts in non-NORMAL state ({startup_state}); "
                        "clear/reset it before enabling or moving"
                  )

      if ENABLE_MOTORS:
            joint_states = robot.get_joints_enable_status_list()
            print("Joint states before enable:", joint_states)
            if all(joint_states):
                  enabled = True
                  print("All joints are already enabled; skipping enable command")
            else:
                  print("Enabling all motor joints...")
                  enable_deadline = time.monotonic() + ENABLE_TIMEOUT
                  enabled = False
                  reconnect_attempts = 0
                  while time.monotonic() < enable_deadline:
                        try:
                              enabled = robot.enable()
                        except RuntimeError as exc:
                              if "Failed to send" not in str(exc):
                                    raise
                              if reconnect_attempts >= ENABLE_RECONNECT_ATTEMPTS:
                                    raise
                              print(
                                    "Enable transmit failed; reopening CAN adapter..."
                              )
                              disconnect_nero(robot, settle_time=0.5)
                              time.sleep(0.5)
                              robot.connect()
                              reconnect_attempts += 1
                              continue
                        if enabled and all(robot.get_joints_enable_status_list()):
                              break
                        time.sleep(0.05)
                  print("All joints enabled:", enabled)
      else:
            print("Motor enable command disabled; checking that all joints are enabled")

      if ENABLE_MOTORS:
            print("Waiting for arm state NORMAL after enable...")
            wait_for_normal_state(robot)
      elif startup_status.msg.arm_status != robot.ARM_STATUS.ArmStatus.NORMAL:
            raise RuntimeError(
                  f"Arm remains in non-NORMAL state ({startup_status.msg.arm_status}); "
                  "enable the motors before moving"
            )

      time.sleep(0.5)

      print("Communication OK:", robot.is_ok())

      if not robot.is_ok():
            raise RuntimeError("Arm is not OK; refusing to move")

      if not all(robot.get_joints_enable_status_list()):
            raise RuntimeError("Not all joints are enabled; refusing to move")

      status = robot.get_arm_status()
      if status is None:
            raise RuntimeError("No arm status available; refusing to move")
      if status.msg.arm_status != robot.ARM_STATUS.ArmStatus.NORMAL:
            arm_state = status.msg.arm_status
            if arm_state == robot.ARM_STATUS.ArmStatus.NO_SOLUTION:
                  detail = (
                        "NO_SOLUTION(0x2) means the previous/requested target "
                        "has no valid inverse-kinematic solution; inspect the "
                        "target and current pose, then reset before moving"
                  )
            elif arm_state == robot.ARM_STATUS.ArmStatus.EMERGENCY_STOP:
                  detail = (
                        "release the physical/UI emergency stop, then reset "
                        "before moving"
                  )
            else:
                  detail = "clear/reset the reported arm state before moving"
            raise RuntimeError(
                  f"Arm is not in NORMAL state ({arm_state}); {detail}"
            )

      firmware = robot.get_firmware()
      print("Firmware:", firmware["software_version"] if firmware else None)

      joints = robot.get_joint_angles()
      if joints is None:
            raise RuntimeError("No valid joint-angle feedback; refusing to move")
      current_joints = joints.msg.copy()
      target_joints = current_joints.copy()
      target_joints[MOVE_JOINT_INDEX] += MOVE_JOINT_DELTA_RAD
      print("Current joint angles [rad]:", current_joints)
      print("Target joint angles [rad]:", target_joints)

      if PLATFORM_TOP_Z is None:
            raise RuntimeError(
                  "Set PLATFORM_TOP_Z to the measured platform height "
                  "in the Nero base coordinate frame before moving."
            )

      pose = robot.get_flange_pose()
      if pose is not None:
            minimum_flange_z = PLATFORM_TOP_Z + SAFETY_MARGIN + TOOL_LOWEST_POINT
            print("Current flange pose [m, rad]:", pose.msg)
            print("Minimum allowed current flange z [m]:", minimum_flange_z)
            if pose.msg[2] < minimum_flange_z:
                  raise RuntimeError(
                        f"Current flange z={pose.msg[2]:.3f} m is below the "
                        f"safe floor {minimum_flange_z:.3f} m"
                  )

      robot.set_speed_percent(SPEED_PERCENT)
      print(
            f"Planned joint move: joint {MOVE_JOINT_INDEX + 1}, "
            f"{MOVE_JOINT_DELTA_RAD:.4f} rad"
      )
      input(
            f"Press Enter to move at {SPEED_PERCENT}% speed, "
            "or Ctrl-C to abort: "
      )
      robot.move_j(target_joints)
      wait_motion_done(robot)
      print("Motion completed")

finally:
      if connected:
            if DISABLE_ON_EXIT:
                  try:
                        print("Disabling all motor joints before disconnect...")
                        robot.disable()
                        time.sleep(0.5)
                  except Exception as exc:
                        print(f"Warning: could not disable joints cleanly: {exc}")
            disconnect_nero(robot)
            print("Disconnected")
