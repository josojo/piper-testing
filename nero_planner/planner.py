"""Bounded pose planning with continuous conservative clearance certificates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
import uuid

import mujoco
import numpy as np

from .model import JOINT_NAMES, Scene


class PlanningError(ValueError):
    """An input, motion, or playback precondition was rejected."""


def vector(value, length, name):
    if not isinstance(value, (list, tuple, np.ndarray)) or len(value) != length:
        raise PlanningError(f"{name} must contain {length} finite numbers")
    if any(isinstance(v, (bool, str)) or not isinstance(v, (int, float, np.number)) for v in value):
        raise PlanningError(f"{name} must contain {length} finite numbers")
    array = np.asarray(value, dtype=float)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise PlanningError(f"{name} must contain {length} finite numbers")
    return array


@dataclass(frozen=True)
class Pose:
    frame: str
    position_m: tuple[float, ...]
    orientation_xyzw: tuple[float, ...]
    gripper: float = 1.0
    reason: str = ""

    def __post_init__(self):
        if self.frame != "nero_base":
            raise PlanningError("Only the nero_base frame is supported")
        position = vector(self.position_m, 3, "position_m")
        quat = vector(self.orientation_xyzw, 4, "orientation_xyzw")
        norm = np.linalg.norm(quat)
        if not np.isfinite(norm) or abs(norm - 1) > 1e-3:
            raise PlanningError("orientation_xyzw must be a unit quaternion")
        if isinstance(self.gripper, bool) or self.gripper != 1.0:
            raise PlanningError("Only the fully open gripper (gripper=1.0) is supported")
        if not isinstance(self.reason, str):
            raise PlanningError("reason must be text")
        object.__setattr__(self, "position_m", tuple(position))
        object.__setattr__(self, "orientation_xyzw", tuple(quat / norm))

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise PlanningError("Target must be a JSON object")
        required = {"frame", "position_m", "orientation_xyzw"}
        if not required <= value.keys() or value.keys() - (required | {"gripper", "reason"}):
            raise PlanningError("Target requires frame, position_m, orientation_xyzw; optional gripper and reason")
        return cls(**value)


@dataclass(frozen=True)
class Limits:
    minimum_clearance_m: float = 0.03
    max_velocity_rad_s: float = 0.2
    max_acceleration_rad_s2: float = 0.5
    max_joint_displacement_rad: float = 0.25
    max_target_translation_m: float = 0.05
    position_tolerance_m: float = 0.002
    orientation_tolerance_rad: float = math.radians(2)
    max_duration_s: float = 15.0
    sample_period_s: float = 0.02
    max_subdivision_depth: int = 16
    max_validation_samples: int = 20000
    max_ik_iterations: int = 400

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise PlanningError(f"{name} must be finite and positive")
        for name in ("max_subdivision_depth", "max_validation_samples", "max_ik_iterations"):
            if not isinstance(getattr(self, name), int):
                raise PlanningError(f"{name} must be an integer")
        if self.max_subdivision_depth > 24:
            raise PlanningError("max_subdivision_depth must be at most 24")


@dataclass(frozen=True)
class SimulatedState:
    qpos: tuple[float, ...]
    captured_at: float
    scene_fingerprint: str


@dataclass(frozen=True)
class ValidationResult:
    status: str
    minimum_clearance_m: float
    position_error_m: float
    orientation_error_rad: float
    checked_configurations: int
    peak_velocity_rad_s: float
    peak_acceleration_rad_s2: float


@dataclass(frozen=True)
class ValidatedTrajectory:
    joint_names: tuple[str, ...]
    positions: tuple[tuple[float, ...], ...]
    timestamps: tuple[float, ...]
    source_target_pose: Pose
    starting_state: SimulatedState
    validation: ValidationResult
    limits: Limits
    validation_id: str

    def to_dict(self):
        return asdict(self)


def rotation_error(target_wxyz, current_wxyz):
    inverse, relative, error = np.empty(4), np.empty(4), np.empty(3)
    mujoco.mju_negQuat(inverse, current_wxyz)
    mujoco.mju_mulQuat(relative, target_wxyz, inverse)
    if relative[0] < 0:
        relative *= -1
    mujoco.mju_quat2Vel(error, relative, 1.0)
    return error


def smoothstep(alpha):
    return alpha ** 3 * (10 + alpha * (-15 + 6 * alpha))


class Planner:
    def __init__(self, scene_path, limits=None):
        self.scene = Scene(scene_path)
        self.limits = limits or Limits()
        self._issued = {}
        self._model_fingerprint = self.scene.fingerprint()

    def set_start(self, q):
        q = self._joints(q)
        self.scene.set_arm(q)

    def _joints(self, q):
        q = vector(q, 7, "joint configuration")
        if np.any(q < self.scene.ranges[:, 0]) or np.any(q > self.scene.ranges[:, 1]):
            raise PlanningError("Joint-position limit exceeded")
        return q

    def current_pose(self):
        # Observation must not reset velocity or silently repair the gripper;
        # those are preconditions that plan() must be able to reject.
        mujoco.mj_forward(self.scene.model, self.scene.data)
        position = self.scene.data.site_xpos[self.scene.site].copy()
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.scene.data.site_xmat[self.scene.site])
        return Pose("nero_base", tuple(position), tuple(quat[[1, 2, 3, 0]]))

    def _errors(self, q, target):
        position, quat = self.scene.pose(q)
        target_quat = np.asarray(target.orientation_xyzw)[[3, 0, 1, 2]]
        return np.asarray(target.position_m) - position, rotation_error(target_quat, quat)

    def _solve_ik(self, start, target):
        q = start.copy()
        lower = np.maximum(self.scene.ranges[:, 0], start - self.limits.max_joint_displacement_rad)
        upper = np.minimum(self.scene.ranges[:, 1], start + self.limits.max_joint_displacement_rad)
        jp, jr = np.zeros((3, self.scene.model.nv)), np.zeros((3, self.scene.model.nv))
        weights = np.array([1, 1, 1, 0.2, 0.2, 0.2])
        for _ in range(self.limits.max_ik_iterations):
            ep, er = self._errors(q, target)
            if np.linalg.norm(ep) <= self.limits.position_tolerance_m and np.linalg.norm(er) <= self.limits.orientation_tolerance_rad:
                return q
            mujoco.mj_jacSite(self.scene.model, self.scene.data, jp, jr, self.scene.site)
            jac = np.vstack((jp[:, self.scene.dadr], jr[:, self.scene.dadr])) * weights[:, None]
            error = np.concatenate((ep, er)) * weights
            inverse = jac.T @ np.linalg.solve(jac @ jac.T + 1e-5 * np.eye(6), np.eye(6))
            delta = inverse @ error + 0.02 * (np.eye(7) - inverse @ jac) @ (start - q)
            delta *= min(1.0, 0.05 / max(np.max(np.abs(delta)), 1e-12))
            improved = False
            for scale in (1.0, 0.5, 0.25, 0.125, 0.0625):
                candidate = np.clip(q + scale * delta, lower, upper)
                cp, cr = self._errors(candidate, target)
                if np.linalg.norm(np.concatenate((cp, cr)) * weights) < np.linalg.norm(error) - 1e-12:
                    q, improved = candidate, True
                    break
            if not improved:
                break
        raise PlanningError("IK failed: target is unreachable within pose tolerances and bounded joint motion")

    def _clearance(self, q):
        distances = self.scene.distances(self._joints(q))
        index = int(np.argmin(distances))
        value = float(distances[index])
        if value < self.limits.minimum_clearance_m:
            a, b = self.scene.pairs[index]
            raise PlanningError(f"Forbidden clearance: {self.scene.model.geom(a).name} / "
                                f"{self.scene.model.geom(b).name}: {value:.6f} m "
                                f"< {self.limits.minimum_clearance_m:.6f} m")
        return value

    def _certify_segment(self, start, goal):
        """Certify the entire straight joint path, not only discrete samples.

        At a midpoint, either geom moves <= R*sum(|dq|)/2 over the interval.
        Subtract both bounds from the midpoint distance lower bound. If that
        fails, bisect; fail closed when the resolution/work limit is reached.
        """
        minimum = min(self._clearance(start), self._clearance(goal))
        count = 2
        stack = [(start, goal, 0)]
        while stack:
            a, b, depth = stack.pop()
            if count >= self.limits.max_validation_samples:
                raise PlanningError("Path clearance could not be certified within the validation budget")
            midpoint = (a + b) / 2
            distance = self._clearance(midpoint)
            count += 1
            bound = self.scene.motion_radius * np.abs(b - a).sum()
            certified = distance - bound
            if certified >= self.limits.minimum_clearance_m:
                minimum = min(minimum, certified)
            elif depth >= self.limits.max_subdivision_depth:
                raise PlanningError("Path clearance could not be certified between samples")
            else:
                stack.extend(((midpoint, b, depth + 1), (a, midpoint, depth + 1)))
        return float(minimum), count

    def plan(self, target: Pose):
        if not isinstance(target, Pose):
            raise PlanningError("plan requires a validated Pose")
        if self.scene.fingerprint() != self._model_fingerprint:
            raise PlanningError("Scene was modified; create a new planner before planning")
        saved_qpos = self.scene.data.qpos.copy()
        saved_qvel = self.scene.data.qvel.copy()
        state = SimulatedState(tuple(saved_qpos), time.monotonic(), self.scene.fingerprint())
        try:
            if not np.isfinite(saved_qpos).all() or not np.isfinite(saved_qvel).all() or np.max(np.abs(saved_qvel)) > 1e-9:
                raise PlanningError("Starting state must be finite and stationary")
            if any(abs(saved_qpos[address] - value) > 1e-9 for address, value in self.scene.gripper.items()):
                raise PlanningError("Starting gripper must be fully open and synchronized")
            start = self._joints(saved_qpos[self.scene.qadr])
            self._clearance(start)
            position, _ = self.scene.pose(start)
            if np.linalg.norm(np.asarray(target.position_m) - position) > self.limits.max_target_translation_m + 1e-12:
                raise PlanningError("Target translation exceeds the per-request step limit")
            goal = self._solve_ik(start, target)
            return self._validate(start, goal, target, state)
        finally:
            self.scene.data.qpos[:] = saved_qpos
            self.scene.data.qvel[:] = saved_qvel
            mujoco.mj_forward(self.scene.model, self.scene.data)

    def _validate(self, start, goal, target, state):
        self._joints(goal)
        displacement = float(np.max(np.abs(goal - start)))
        if displacement > self.limits.max_joint_displacement_rad + 1e-12:
            raise PlanningError("Maximum joint displacement exceeded")
        # Exact maxima of derivatives of s(u)=10u³-15u⁴+6u⁵ on [0,1].
        duration = max(self.limits.sample_period_s,
                       1.875 * displacement / self.limits.max_velocity_rad_s,
                       math.sqrt((10 / math.sqrt(3)) * displacement / self.limits.max_acceleration_rad_s2))
        if duration > self.limits.max_duration_s:
            raise PlanningError("Expected execution duration exceeds the limit")
        sample_intervals = duration / self.limits.sample_period_s
        if not math.isfinite(sample_intervals) or sample_intervals > self.limits.max_validation_samples - 1:
            raise PlanningError("Trajectory sample count exceeds the validation budget")
        ep, er = self._errors(goal, target)
        pe, re = float(np.linalg.norm(ep)), float(np.linalg.norm(er))
        if pe > self.limits.position_tolerance_m or re > self.limits.orientation_tolerance_rad:
            raise PlanningError("Final end-effector pose error exceeds tolerance")
        minimum, checks = self._certify_segment(start, goal)
        timestamps = np.linspace(0, duration, max(2, math.ceil(sample_intervals) + 1))
        positions = start + smoothstep(timestamps[:, None] / duration) * (goal - start)
        positions[0], positions[-1] = start, goal
        validation = ValidationResult("validated", minimum, pe, re, checks,
                                      1.875 * displacement / duration,
                                      (10 / math.sqrt(3)) * displacement / duration ** 2)
        result = ValidatedTrajectory(JOINT_NAMES, tuple(map(tuple, positions)), tuple(timestamps),
                                     target, state, validation, self.limits, uuid.uuid4().hex)
        self._issued[result.validation_id] = result
        return result

    def prepare_playback(self, trajectory):
        if not isinstance(trajectory, ValidatedTrajectory) or self._issued.get(trajectory.validation_id) != trajectory:
            raise PlanningError("Trajectory was not issued by this planner or has been modified")
        if trajectory.limits != self.limits or trajectory.starting_state.scene_fingerprint != self.scene.fingerprint():
            raise PlanningError("Scene or limits changed since validation; re-plan")
        if (not np.allclose(self.scene.data.qpos, trajectory.starting_state.qpos, atol=1e-6, rtol=0)
                or not np.isfinite(self.scene.data.qvel).all() or np.max(np.abs(self.scene.data.qvel)) > 1e-9):
            raise PlanningError("Simulated starting state changed since validation; re-plan")

    def playback(self, trajectory, viewer=False):
        self.prepare_playback(trajectory)
        if not viewer:
            for q in trajectory.positions:
                self.scene.set_arm(q)
            return
        import mujoco.viewer

        with mujoco.viewer.launch_passive(self.scene.model, self.scene.data) as window:
            window.opt.geomgroup[3] = 0
            self.prepare_playback(trajectory)
            started = time.monotonic()
            start, goal = np.asarray(trajectory.positions)[[0, -1]]
            duration = trajectory.timestamps[-1]
            while window.is_running():
                elapsed = min(time.monotonic() - started, duration)
                with window.lock():
                    self.scene.set_arm(start + smoothstep(elapsed / duration) * (goal - start))
                window.sync()
                if elapsed >= duration:
                    break
                time.sleep(min(self.limits.sample_period_s, duration - elapsed))
