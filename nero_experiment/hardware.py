"""Opt-in pyAgxArm adapter for bounded, supervised stop-and-settle moves.

No fast JS/MIT streaming: move_j has its own timing. Independent-joint boxes
are checked before commands, with fresh measured feedback monitored in flight.
This is experimental software, not a real-time or safety-rated controller.
"""

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
import math
import time

import mujoco
import numpy as np

from nero_planner import PlanningError, Pose
from nero_planner.planner import vector


@dataclass(frozen=True)
class HardwareConfig:
    scene_fingerprint: str
    firmware_v121_verified: bool
    joint_conventions_verified: bool
    upright_reference_verified: bool
    tool_and_scene_verified: bool
    physical_estop_tested: bool
    empty_gripper_verified: bool
    upright_joints_rad: tuple
    flange_position_in_link7_m: tuple
    flange_orientation_in_link7_xyzw: tuple

    @classmethod
    def from_dict(cls, value):
        try:
            config = cls(**value)
        except (TypeError, ValueError):
            raise PlanningError("Hardware config is incomplete or contains unknown fields") from None
        for name in ("firmware_v121_verified", "joint_conventions_verified", "upright_reference_verified",
                     "tool_and_scene_verified", "physical_estop_tested", "empty_gripper_verified"):
            if getattr(config, name) is not True:
                raise PlanningError(f"Hardware configuration requires a completed physical check: {name}")
        upright = vector(config.upright_joints_rad, 7, "upright_joints_rad")
        flange = Pose("nero_base", config.flange_position_in_link7_m, config.flange_orientation_in_link7_xyzw)
        if not isinstance(config.scene_fingerprint, str) or len(config.scene_fingerprint) != 64:
            raise PlanningError("Set scene_fingerprint to the reviewed scene's fingerprint from the dry-run report")
        return replace(config, upright_joints_rad=tuple(upright),
                       flange_position_in_link7_m=flange.position_m,
                       flange_orientation_in_link7_xyzw=flange.orientation_xyzw)


@dataclass(frozen=True)
class ArmState:
    joints_rad: tuple
    velocities_rad_s: tuple
    enabled: tuple
    arm_status: int
    motion_status: int
    gripper_width_m: float
    flange_pose_m_rad: tuple
    feedback_timestamps: tuple
    captured_at_unix: float

    def to_dict(self):
        return {"source": "hardware_feedback", **asdict(self)}


MAX_FEEDBACK_AGE_S = 0.25
MAX_FEEDBACK_SKEW_S = 0.15
START_MATCH_RAD = 0.001
SETTLE_TOLERANCE_RAD = 0.0005
TRACKING_ENVELOPE_RAD = 0.002
MICROSTEP_RAD = 0.005
MAX_MEASURED_VELOCITY_RAD_S = 0.10
SPEED_PERCENT = 3


class NeroHardware:
    def __init__(self, robot, gripper, clock=time.monotonic, wall_clock=time.time, sleep=time.sleep):
        self.robot, self.gripper = robot, gripper
        self.clock, self.wall_clock, self.sleep = clock, wall_clock, sleep

    def read(self, require_enabled=False):
        timestamps = []

        def fresh(packet, name):
            if packet is None:
                raise PlanningError(f"Missing {name} feedback")
            packet = deepcopy(packet)  # SDK getters expose mutable cached messages.
            stamp = packet.timestamp
            age = self.wall_clock() - stamp
            if not math.isfinite(stamp) or not 0 <= age <= MAX_FEEDBACK_AGE_S:
                raise PlanningError(f"Stale or invalid {name} feedback timestamp")
            timestamps.append(stamp)
            return packet.msg

        # pyAgxArm 1.0.0's aggregate get_joint_angles can return partial zeroes
        # and only the last group's timestamp. Inspect all four groups instead.
        parser = getattr(self.robot, "_parser", None)
        if parser is None:
            raise PlanningError("Unsupported SDK: per-packet joint feedback is unavailable")
        q = []
        for packet_name, indices in (("joint_12", (1, 2)), ("joint_34", (3, 4)),
                                     ("joint_56", (5, 6)), ("joint_7", (7,))):
            packet = fresh(getattr(parser, packet_name, None), packet_name)
            q.extend(getattr(packet, f"joint_{index}") for index in indices)
        q = vector(q, 7, "measured joints")
        if max(timestamps) - min(timestamps) > 0.02 or self.wall_clock() - min(timestamps) > 0.05:
            raise PlanningError("Joint position packets must be within 20 ms of each other and at most 50 ms old")
        status = fresh(self.robot.get_arm_status(), "arm status")
        state_code = int(status.arm_status)
        if state_code not in (0, 6) or status.err_code != 0:
            raise PlanningError(f"Arm reports unsafe state/error: {state_code}/{status.err_code}")
        if int(status.ctrl_mode) not in (0, 1):
            raise PlanningError("Arm must be in standby or CAN mode; other control sources are not supported")
        enabled, velocities = [], []
        for index in range(1, 8):
            driver = fresh(self.robot.get_driver_states(index), f"joint {index} driver")
            foc = driver.foc_status
            for flag in ("voltage_too_low", "motor_overheating", "driver_overcurrent", "driver_overheating",
                         "collision_status", "driver_error_status", "stall_status"):
                if getattr(foc, flag):
                    raise PlanningError(f"Joint {index} driver fault: {flag}")
            enabled.append(bool(foc.driver_enable_status))
            motor = fresh(self.robot.get_motor_states(index), f"joint {index} motor")
            velocities.append(motor.velocity)
        velocities = vector(velocities, 7, "measured velocities")
        if require_enabled and (state_code != 0 or not all(enabled)):
            raise PlanningError("All joints must already be enabled and NORMAL; this experiment never auto-enables")
        grip = fresh(self.gripper.get_gripper_status(), "gripper")
        width = float(grip.value)
        if grip.mode != "width" or not math.isfinite(width) or abs(width - 0.1) > 0.001:
            raise PlanningError("The modeled AGX gripper must report fully open width 0.100 m (+/- 0.001 m)")
        for flag in ("voltage_too_low", "motor_overheating", "driver_overcurrent", "driver_overheating",
                     "sensor_status", "driver_error_status", "homing_status"):
            if getattr(grip.foc_status, flag):
                raise PlanningError(f"Gripper fault: {flag}")
        # Require all three constituent flange packets as well.
        for name in ("end_pose_xy", "end_pose_zrx", "end_pose_ryrz"):
            fresh(getattr(parser, name, None), name)
        flange = fresh(self.robot.get_flange_pose(), "flange pose")
        flange = vector(flange, 6, "measured flange pose")
        if max(timestamps) - min(timestamps) > MAX_FEEDBACK_SKEW_S:
            raise PlanningError("Feedback packet timestamps are too far apart")
        return ArmState(tuple(q), tuple(velocities), tuple(enabled), state_code, int(status.motion_status),
                        width, tuple(flange), tuple(timestamps), self.wall_clock())

    def stationary(self, require_enabled=False, timeout=5.0):
        deadline, stable_since, first = self.clock() + timeout, None, None
        last_error = "Arm has not settled"
        while self.clock() < deadline:
            state = self.read(require_enabled)
            q = np.asarray(state.joints_rad)
            if state.motion_status == 0 and max(abs(v) for v in state.velocities_rad_s) <= 0.01:
                if first is None or np.max(np.abs(q - first)) > SETTLE_TOLERANCE_RAD:
                    first, stable_since = q, self.clock()
                elif self.clock() - stable_since >= 0.3:
                    return state
            else:
                first, stable_since = None, None
            self.sleep(0.02)
        raise PlanningError(last_error + " within timeout")

    def initial_state(self, timeout=5.0):
        # Connections need time to receive every CAN packet. No writes here.
        deadline = self.clock() + timeout
        while True:
            try:
                return self.stationary(timeout=max(0.01, deadline - self.clock()))
            except PlanningError:
                if self.clock() >= deadline:
                    raise
                self.sleep(0.05)

    def stop(self):
        # Do not automatically disable motors or reset: either may release the
        # arm. An electronic stop is best-effort and still needs physical backup.
        self.robot.electronic_emergency_stop()

    def move_and_settle(self, planner, goal, planned_start, minimum_duration=0.1, record=lambda event: None):
        before = self.stationary(require_enabled=True)
        start, goal = np.asarray(before.joints_rad), np.asarray(goal)
        if np.max(np.abs(start - planned_start)) > START_MATCH_RAD:
            raise PlanningError("Hardware state changed from the planned microstep start")
        lower, upper = certify_hardware_box(planner, start, goal)
        began = self.clock()
        stable_since = None
        previous = before
        record({"event": "command_pending", "before": before.to_dict(), "goal_joints_rad": goal.tolist()})
        # Report I/O can block. Refresh once more immediately before the write
        # to CAN rather than assuming the pre-log observation is still current.
        latest = self.read(require_enabled=True)
        if (latest.motion_status != 0 or max(abs(v) for v in latest.velocities_rad_s) > 0.01
                or np.max(np.abs(np.asarray(latest.joints_rad) - start)) > SETTLE_TOLERANCE_RAD):
            raise PlanningError("Arm changed before the motion command; abort and re-plan")
        self.robot.move_j(goal.tolist())
        while self.clock() - began < max(5.0, minimum_duration + 2.0):
            state = self.read(require_enabled=True)
            q = np.asarray(state.joints_rad)
            if np.any(q < lower) or np.any(q > upper):
                raise PlanningError("Measured joints left the certified motion envelope")
            if max(abs(v) for v in state.velocities_rad_s) > MAX_MEASURED_VELOCITY_RAD_S:
                raise PlanningError("Measured motor velocity exceeded experiment limit")
            # Also estimate velocity from the joint packets, independently of
            # the firmware motor-velocity field. Require each group to advance.
            for index, packet_index in enumerate((0, 0, 1, 1, 2, 2, 3)):
                dt = state.feedback_timestamps[packet_index] - previous.feedback_timestamps[packet_index]
                if dt > 0:
                    speed = abs(q[index] - previous.joints_rad[index]) / dt
                    if speed > MAX_MEASURED_VELOCITY_RAD_S + 0.01:
                        raise PlanningError("Measured joint velocity exceeded experiment limit")
            previous = state
            reached = np.max(np.abs(q - goal)) <= SETTLE_TOLERANCE_RAD
            stationary = state.motion_status == 0 and max(abs(v) for v in state.velocities_rad_s) <= 0.01
            if reached and stationary:
                stable_since = self.clock() if stable_since is None else stable_since
                if self.clock() - stable_since >= 0.3 and self.clock() - began >= minimum_duration:
                    record({"event": "settled", "state": state.to_dict()})
                    return state
            else:
                stable_since = None
            self.sleep(0.02)
        raise PlanningError("Hardware move did not reach and settle at its target before timeout")


def certify_hardware_box(planner, start, goal):
    start, goal = vector(start, 7, "hardware start"), vector(goal, 7, "hardware goal")
    if np.max(np.abs(goal - start)) > MICROSTEP_RAD + START_MATCH_RAD + 1e-9:
        raise PlanningError("Hardware microstep exceeds its joint displacement limit")
    lower = np.minimum(start, goal) - TRACKING_ENVELOPE_RAD
    upper = np.maximum(start, goal) + TRACKING_ENVELOPE_RAD
    if np.any(lower < planner.scene.ranges[:, 0]) or np.any(upper > planner.scene.ranges[:, 1]):
        raise PlanningError("Hardware tracking envelope exceeds joint limits")
    saved_qpos, saved_qvel = planner.scene.data.qpos.copy(), planner.scene.data.qvel.copy()
    try:
        clearance = planner.scene.joint_box_clearance(lower, upper)
        if clearance < planner.limits.minimum_clearance_m:
            raise PlanningError(f"Hardware joint envelope has insufficient certified clearance: {clearance:.6f} m")
    finally:
        planner.scene.data.qpos[:] = saved_qpos
        planner.scene.data.qvel[:] = saved_qvel
        mujoco.mj_forward(planner.scene.model, planner.scene.data)
    return lower, upper


def check_calibration(planner, state, config):
    if planner.scene.fingerprint() != config.scene_fingerprint:
        raise PlanningError("Scene differs from the physically reviewed hardware configuration")
    planner.set_start(state.joints_rad)
    # Compare measured SDK flange pose to the reviewed model flange transform.
    body = planner.scene.model.body("link7").id
    rotation = planner.scene.data.xmat[body].reshape(3, 3)
    predicted = planner.scene.data.xpos[body] + rotation @ np.asarray(config.flange_position_in_link7_m)
    local = np.empty(9)
    quat = np.asarray(config.flange_orientation_in_link7_xyzw)[[3, 0, 1, 2]]
    mujoco.mju_quat2Mat(local, quat)
    predicted_rotation = rotation @ local.reshape(3, 3)
    roll, pitch, yaw = state.flange_pose_m_rad[3:]
    cr, sr, cp, sp, cy, sy = math.cos(roll), math.sin(roll), math.cos(pitch), math.sin(pitch), math.cos(yaw), math.sin(yaw)
    measured_rotation = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                                  [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                                  [-sp, cp*sr, cp*cr]])
    position_error = float(np.linalg.norm(predicted - state.flange_pose_m_rad[:3]))
    angle_error = math.acos(float(np.clip((np.trace(predicted_rotation.T @ measured_rotation) - 1) / 2, -1, 1)))
    if position_error > 0.005 or angle_error > math.radians(3):
        raise PlanningError(f"Measured/model flange disagreement: {position_error:.4f} m, {math.degrees(angle_error):.2f} degrees")


def hardware_schedule(planner, plan):
    """Precheck every microstep before the first physical command."""
    schedule = []
    planner.set_start(plan.original_joints)
    try:
        for segment in plan.trajectories:
            planner.prepare_playback(segment)
            start, goal = np.asarray(segment.positions)[[0, -1]]
            count = max(1, math.ceil(float(np.max(np.abs(goal - start))) / MICROSTEP_RAD))
            points = np.linspace(start, goal, count + 1)
            for a, b in zip(points[:-1], points[1:]):
                certify_hardware_box(planner, a, b)
                schedule.append((tuple(a), tuple(b), max(0.1, segment.timestamps[-1] / count)))
                if len(schedule) > 2000:
                    raise PlanningError("Hardware microstep budget exceeded")
            planner.playback(segment)
        return tuple(schedule)
    finally:
        planner.set_start(plan.original_joints)


def execute(planner, plan, hardware, config, confirm, progress=lambda message: None, record=lambda event: None):
    if plan.model == "offline-test-double":
        raise PlanningError("Offline test-double plans cannot execute on hardware")
    if not np.allclose(plan.upright_joints, config.upright_joints_rad, atol=1e-9, rtol=0):
        raise PlanningError("Planned upright reference differs from reviewed hardware configuration")
    schedule = hardware_schedule(planner, plan)
    state = hardware.stationary(require_enabled=True)
    check_calibration(planner, state, config)
    if np.max(np.abs(np.asarray(state.joints_rad) - plan.original_joints)) > START_MATCH_RAD:
        raise PlanningError("Arm moved during LLM planning/preview; capture again and re-plan")
    if not schedule:
        return state
    if not confirm(f"Simulation and {len(schedule)} hardware envelopes passed. Type EXECUTE to move at {SPEED_PERCENT}% speed: "):
        raise PlanningError("Physical execution cancelled")
    # This guard is before speed or motion writes. Repeat after the prompt.
    state = hardware.stationary(require_enabled=True)
    check_calibration(planner, state, config)
    if np.max(np.abs(np.asarray(state.joints_rad) - plan.original_joints)) > START_MATCH_RAD:
        raise PlanningError("Arm changed while awaiting confirmation; re-plan")
    started = hardware.clock()
    try:
        hardware.robot.set_speed_percent(SPEED_PERCENT)
        for index, (start, goal, duration) in enumerate(schedule):
            if hardware.clock() - started > 1200:
                raise PlanningError("Overall hardware execution timeout exceeded")
            if planner.scene.fingerprint() != config.scene_fingerprint:
                raise PlanningError("Scene changed during execution")
            state = hardware.move_and_settle(planner, goal, start, duration, record)
            check_calibration(planner, state, config)
            progress(f"Hardware microstep {index + 1}/{len(schedule)} settled")
        if np.max(np.abs(np.asarray(state.joints_rad) - plan.original_joints)) > START_MATCH_RAD:
            raise PlanningError("Hardware return did not reach the original joints")
        return state
    except BaseException as error:
        try:
            hardware.stop()
        except Exception:
            raise RuntimeError("Execution failed AND the electronic stop could not be sent; use the physical emergency stop") from error
        raise


@contextmanager
def connect():
    # Reading and offline imports never load pyAgxArm. Private packet access is
    # pinned to the SDK whose aggregate-feedback behavior was inspected.
    if version("pyAgxArm") != "1.0.0":
        raise PlanningError("This adapter requires pyAgxArm==1.0.0; review packet access before changing versions")
    from nero_safety_common import connect_nero, disconnect_nero
    robot = connect_nero()
    try:
        gripper = robot.init_effector("agx_gripper")
        yield NeroHardware(robot, gripper)
    finally:
        disconnect_nero(robot)
