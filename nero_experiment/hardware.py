"""Opt-in pyAgxArm adapter for bounded, supervised stop-and-settle moves.

No fast JS/MIT streaming: move_j has its own timing. Independent-joint boxes
are checked before commands, with fresh measured feedback monitored in flight.
This is experimental software, not a real-time or safety-rated controller.
"""

from contextlib import contextmanager
from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass, replace
import math
import time

import mujoco
import numpy as np

from nero_planner import PlanningError, Pose
from nero_planner.planner import vector
from .sdk_compat import (acceleration_counts, check_sdk_revision,
                         read_joint_acceleration, write_joint_acceleration)


@dataclass(frozen=True)
class HardwareConfig:
    scene_fingerprint: str
    firmware_v121_verified: bool
    joint_conventions_verified: bool
    upright_reference_verified: bool
    tool_and_scene_verified: bool
    physical_estop_tested: bool
    upright_joints_rad: tuple
    flange_position_in_link7_m: tuple
    flange_orientation_in_link7_xyzw: tuple

    @classmethod
    def from_dict(cls, value):
        # Accept the old example/local config shape after removing its gripper gate.
        if not isinstance(value, dict):
            raise PlanningError("Hardware config must be a JSON object")
        value = dict(value)
        value.pop("empty_gripper_verified", None)
        try:
            config = cls(**value)
        except (TypeError, ValueError):
            raise PlanningError("Hardware config is incomplete or contains unknown fields") from None
        for name in ("firmware_v121_verified", "joint_conventions_verified", "upright_reference_verified",
                     "tool_and_scene_verified", "physical_estop_tested"):
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
MICROSTEP_RAD = 0.002
MAX_MEASURED_VELOCITY_RAD_S = 0.10
SPEED_PERCENT = 1
MAX_CONTROLLER_ACCELERATION_RAD_S2 = 0.15
VELOCITY_HISTORY_SAMPLES = 26  # About 0.5 s at the nominal 20 ms polling interval.


class VelocityLimitError(PlanningError):
    """Keep feedback in memory until execute has attempted the emergency stop."""

    def __init__(self, reason, diagnostic):
        super().__init__(reason)
        self.diagnostic = diagnostic


class JointFeedbackTimingError(PlanningError):
    """A snapshot failed the joint age/skew limits; never use its positions."""


class NeroHardware:
    def __init__(self, robot, clock=time.monotonic, wall_clock=time.time, sleep=time.sleep):
        self.robot = robot
        self.clock, self.wall_clock, self.sleep = clock, wall_clock, sleep

    def configure_acceleration(self, maximum=MAX_CONTROLLER_ACCELERATION_RAD_S2, record=lambda event: None):
        """Lower limits without motion; never restore higher limits automatically."""
        maximum = min(maximum, MAX_CONTROLLER_ACCELERATION_RAD_S2)
        cap = acceleration_counts(maximum) / 100
        firmware = self.robot.get_firmware()
        if not isinstance(firmware, dict) or firmware.get("software_version") != "1.21":
            raise PlanningError("The acceleration compatibility fix requires confirmed NERO firmware 1.21")
        self.stationary()
        # Read every original value before making any changes.
        original = [read_joint_acceleration(self.robot, j, self.wall_clock) for j in range(1, 8)]
        targets = [acceleration_counts(min(value, cap)) / 100 for value in original]
        record({"event": "controller_acceleration_pending", "firmware": firmware,
                "previous_rad_s2": original, "requested_rad_s2": targets,
                "limits_retained_after_execution": True})
        self.stationary()
        for joint, (before, target) in enumerate(zip(original, targets), 1):
            if not math.isclose(before, target, abs_tol=1e-9, rel_tol=0):
                write_joint_acceleration(self.robot, joint, target, self.wall_clock, self.clock, self.sleep)
            record({"event": "joint_acceleration_verified", "joint": joint,
                    "previous_rad_s2": before, "applied_rad_s2": target})
        # Final independent fresh read of all seven joints before permitting motion.
        actual = [read_joint_acceleration(self.robot, j, self.wall_clock) for j in range(1, 8)]
        if any(not math.isclose(a, b, abs_tol=1e-9, rel_tol=0) for a, b in zip(actual, targets)):
            raise PlanningError("Controller acceleration limits changed during setup; motion prohibited")
        record({"event": "controller_acceleration_configured", "applied_rad_s2": actual})
        return actual

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
        joint_timestamps = tuple(timestamps)
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
        # Require all three constituent flange packets as well.
        for name in ("end_pose_xy", "end_pose_zrx", "end_pose_ryrz"):
            fresh(getattr(parser, name, None), name)
        flange = fresh(self.robot.get_flange_pose(), "flange pose")
        flange = vector(flange, 6, "measured flange pose")
        if max(timestamps) - min(timestamps) > MAX_FEEDBACK_SKEW_S:
            raise PlanningError("Feedback packet timestamps are too far apart")
        # Check at the end so driver/controller faults take precedence and time
        # spent gathering other feedback counts toward snapshot age.
        ages = [self.wall_clock() - stamp for stamp in joint_timestamps]
        skew = max(joint_timestamps) - min(joint_timestamps)
        if skew > 0.02 or max(ages) > 0.05:
            detail = ", ".join(f"{name}={age * 1000:.1f} ms" for name, age in
                               zip(("joint_12", "joint_34", "joint_56", "joint_7"), ages))
            raise JointFeedbackTimingError(
                f"Joint feedback timing rejected: skew={skew * 1000:.1f} ms (limit 20 ms); "
                f"ages [{detail}] (limit 50 ms)")
        return ArmState(tuple(q), tuple(velocities), tuple(enabled), state_code, int(status.motion_status),
                        tuple(flange), tuple(timestamps), self.wall_clock())

    def stationary(self, require_enabled=False, timeout=5.0):
        deadline, stable_since, first = self.clock() + timeout, None, None
        last_error = "Arm has not settled"
        while self.clock() < deadline:
            try:
                state = self.read(require_enabled)
            except JointFeedbackTimingError as error:
                # Only before commands / after a previously settled move.
                # A rejected snapshot cannot contribute to the settling dwell.
                stable_since, first = None, None
                last_error = str(error)
                self.sleep(0.005)
                continue
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
        latest = self.stationary(require_enabled=True)
        if (latest.motion_status != 0 or max(abs(v) for v in latest.velocities_rad_s) > 0.01
                or np.max(np.abs(np.asarray(latest.joints_rad) - start)) > SETTLE_TOLERANCE_RAD):
            raise PlanningError("Arm changed before the motion command; abort and re-plan")
        previous = latest
        began = self.clock()
        samples = deque([latest], maxlen=VELOCITY_HISTORY_SAMPLES)

        def velocity_failure(reason, source, joint, velocity, limit):
            return VelocityLimitError(reason, {
                "event": "velocity_limit_exceeded", "reason": reason,
                "source": source, "joint": joint, "velocity_rad_s": float(velocity),
                "limit_rad_s": limit, "goal_joints_rad": goal.tolist(),
                "elapsed_since_command_s": self.clock() - began,
                "samples": tuple(samples),
            })

        self.robot.move_j(goal.tolist())
        while self.clock() - began < max(5.0, minimum_duration + 2.0):
            state = self.read(require_enabled=True)
            samples.append(state)
            q = np.asarray(state.joints_rad)
            if np.any(q < lower) or np.any(q > upper):
                raise PlanningError("Measured joints left the certified motion envelope")
            motor_speed = max(abs(v) for v in state.velocities_rad_s)
            if motor_speed > MAX_MEASURED_VELOCITY_RAD_S:
                joint = 1 + max(range(7), key=lambda i: abs(state.velocities_rad_s[i]))
                reason = (f"Measured motor velocity exceeded experiment limit: joint {joint} "
                                    f"{state.velocities_rad_s[joint - 1]:.4f} rad/s > "
                                    f"{MAX_MEASURED_VELOCITY_RAD_S:.4f} rad/s")
                raise velocity_failure(reason, "motor_feedback", joint,
                                       state.velocities_rad_s[joint - 1], MAX_MEASURED_VELOCITY_RAD_S)
            # Also estimate velocity from the joint packets, independently of
            # the firmware motor-velocity field. Require each group to advance.
            for index, packet_index in enumerate((0, 0, 1, 1, 2, 2, 3)):
                dt = state.feedback_timestamps[packet_index] - previous.feedback_timestamps[packet_index]
                if dt > 0:
                    speed = abs(q[index] - previous.joints_rad[index]) / dt
                    if speed > MAX_MEASURED_VELOCITY_RAD_S + 0.01:
                        reason = (f"Measured joint velocity exceeded experiment limit: joint {index + 1} "
                                            f"{speed:.4f} rad/s > "
                                            f"{MAX_MEASURED_VELOCITY_RAD_S + 0.01:.4f} rad/s")
                        raise velocity_failure(reason, "joint_finite_difference", index + 1,
                                               speed, MAX_MEASURED_VELOCITY_RAD_S + 0.01)
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
    acceleration_cap = min(planner.limits.max_acceleration_rad_s2, MAX_CONTROLLER_ACCELERATION_RAD_S2)
    if not confirm(f"Simulation and {len(schedule)} hardware envelopes passed. Type EXECUTE to apply "
                   f"a {acceleration_cap:g} rad/s² acceleration cap (retained afterward) "
                   f"and move at {SPEED_PERCENT}% speed: "):
        raise PlanningError("Physical execution cancelled")
    # This guard is before speed or motion writes. Repeat after the prompt.
    state = hardware.stationary(require_enabled=True)
    check_calibration(planner, state, config)
    if np.max(np.abs(np.asarray(state.joints_rad) - plan.original_joints)) > START_MATCH_RAD:
        raise PlanningError("Arm changed while awaiting confirmation; re-plan")
    started = hardware.clock()
    try:
        applied = hardware.configure_acceleration(acceleration_cap, record)
        progress(f"Verified controller acceleration limits: {applied} rad/s²")
        state = hardware.stationary(require_enabled=True)
        if np.max(np.abs(np.asarray(state.joints_rad) - plan.original_joints)) > START_MATCH_RAD:
            raise PlanningError("Arm changed during acceleration setup; re-plan")
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
        finally:
            if isinstance(error, VelocityLimitError):
                # Serialization and report I/O must never delay the stop attempt
                # or obscure its failure. No per-sample disk writes in motion.
                try:
                    record({**error.diagnostic,
                            "samples": [state.to_dict() for state in error.diagnostic["samples"]]})
                except Exception:
                    pass  # Preserve the motion/stop error if reporting fails.
        raise


@contextmanager
def connect():
    # Reading and offline imports never load pyAgxArm. Private packet access is
    # pinned to the SDK whose aggregate-feedback behavior was inspected.
    check_sdk_revision()
    from nero_safety_common import connect_nero, disconnect_nero
    robot = connect_nero()
    try:
        yield NeroHardware(robot)
    finally:
        disconnect_nero(robot)
