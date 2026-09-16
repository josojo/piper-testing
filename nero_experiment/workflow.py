"""Plan a complete bounded upright-and-return experiment before execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time

import numpy as np

from nero_planner import Limits, Planner, PlanningError, ValidatedTrajectory
from nero_planner.model import JOINT_NAMES
from nero_planner.planner import vector
from .openrouter import validate_selection


EXPERIMENT_LIMITS = Limits(minimum_clearance_m=0.005, max_velocity_rad_s=0.08,
                          max_acceleration_rad_s2=0.15, max_duration_s=30.0,
                          max_joint_displacement_rad=1.0, max_target_translation_m=0.5)
DEMO_START = (0.0, -0.15, 0.0, 0.30, 0.0, -0.15, 0.0)
UPRIGHT = (0.0,) * 7


@dataclass(frozen=True)
class Candidate:
    id: str
    trajectory: ValidatedTrajectory
    remaining_joint_error_rad: float

    def to_prompt(self):
        pose = asdict(self.trajectory.source_target_pose)
        pose.pop("reason")
        pose.pop("gripper")
        return {"id": self.id, "target": pose,
                "remaining_joint_error_rad": self.remaining_joint_error_rad,
                "validation": asdict(self.trajectory.validation),
                "simulation_duration_s": self.trajectory.timestamps[-1]}


@dataclass(frozen=True)
class ExperimentPlan:
    original_joints: tuple
    upright_joints: tuple
    trajectories: tuple[ValidatedTrajectory, ...]
    outward_count: int
    decisions: tuple[dict, ...]
    model: str
    captured_state: dict

    def to_dict(self):
        return {"status": "simulation_passed", "hardware_executed": False,
                "task": "straighten_upright_then_return", "model": self.model,
                "captured_state": self.captured_state,
                "original_joints_rad": self.original_joints,
                "upright_joints_rad": self.upright_joints,
                "outward_segments": self.outward_count,
                "return_segments": len(self.trajectories) - self.outward_count,
                "decisions": self.decisions,
                "trajectories": [t.to_dict() for t in self.trajectories],
                "note": "Report only; cannot be loaded as authorization to move hardware."}


def reachable_candidates(planner, upright, remaining_steps=3):
    current = planner.scene.data.qpos[planner.scene.qadr].copy()
    delta = np.asarray(upright) - current
    distance = float(np.max(np.abs(delta)))
    if distance <= 1e-9:
        return []
    max_step = planner.limits.max_joint_displacement_rad
    minimum_step = max(0.0, distance - (remaining_steps - 1) * max_step)
    largest_step = min(distance, max_step)
    if minimum_step > largest_step + 1e-12:
        raise PlanningError(f"Cannot reach upright within {remaining_steps} pose requests at the configured joint-step limit")
    step_sizes = sorted({largest_step,
                         max(minimum_step, largest_step * 0.75),
                         max(minimum_step, largest_step * 0.5),
                         max(minimum_step, largest_step * 0.25)}, reverse=True)
    result = []
    for step in step_sizes:
        goal = current + (step / distance) * delta
        try:
            trajectory = planner.plan_joint_goal(goal, "Slow, validated waypoint toward the upright joint reference")
        except PlanningError:
            continue
        result.append(Candidate(f"candidate_{len(result) + 1}", trajectory,
                                float(np.max(np.abs(np.asarray(upright) - goal)))))
    if not result:
        raise PlanningError("No reachable, collision-free step toward upright; no hardware motion allowed")
    return result


def plan_experiment(planner: Planner, start, chooser, upright=UPRIGHT, captured_state=None,
                    max_steps=3, progress=lambda message: None):
    start = vector(start, 7, "start")
    upright = vector(upright, 7, "upright")
    if not 1 <= max_steps <= 64:
        raise PlanningError("max-steps must be between 1 and 64")
    if np.max(np.abs(upright - start)) > 3.0:
        raise PlanningError("Initial experiment exceeds the 3 rad per-joint total excursion budget")
    required_steps = math.ceil(float(np.max(np.abs(upright - start))) /
                               planner.limits.max_joint_displacement_rad - 1e-12)
    if required_steps > max_steps:
        raise PlanningError(f"Upright posture needs at least {required_steps} pose requests at the configured joint-step limit")
    original_state = captured_state or {"source": "simulation", "joints_rad": start.tolist()}
    planner.set_start(upright)
    # Both endpoints must be valid even before paying for LLM requests.
    planner.plan_joint_goal(upright)
    planner.set_start(start)
    planner.plan_joint_goal(start)
    segments, decisions = [], []
    visited = [tuple(start)]
    try:
        while np.max(np.abs(planner.scene.data.qpos[planner.scene.qadr] - upright)) > 1e-9:
            if len(segments) >= max_steps:
                raise PlanningError("Upright planning exceeded its step/API request budget")
            remaining_steps = max_steps - len(segments)
            candidates = reachable_candidates(planner, upright, remaining_steps)
            context = {
                "task": "Straighten the arm upright, then move slowly back to the original joint configuration",
                "initial_observation": original_state,
                "current_state_source": "simulation_predicted_from_initial_observation",
                "current_joints_rad": planner.scene.data.qpos[planner.scene.qadr].tolist(),
                "current_tool_pose": asdict(planner.current_pose()),
                "upright_joints_rad": upright.tolist(), "original_joints_rad": start.tolist(),
                "pose_requests_remaining": remaining_steps,
                "joint_names": list(JOINT_NAMES),
                "joint_limits_rad": planner.scene.ranges.tolist(),
                "limits": asdict(planner.limits),
            }
            correction = None
            for attempt in range(3):
                progress(f"Requesting next upright pose ({len(segments) + 1}, attempt {attempt + 1})")
                try:
                    response = chooser.choose(context, candidates, correction)
                    chosen = validate_selection(response, candidates)
                    break
                except PlanningError as error:
                    correction = str(error)
            else:
                raise PlanningError("LLM repeatedly returned invalid coordinates: " + correction)
            planner.playback(chosen.trajectory)
            segments.append(chosen.trajectory)
            decisions.append(response)
            visited.append(chosen.trajectory.positions[-1])
            progress(f"Simulated upright step {len(segments)}; remaining joint error {chosen.remaining_joint_error_rad:.4f} rad")
        outward_count = len(segments)
        for joints in reversed(visited[:-1]):
            trajectory = planner.plan_joint_goal(joints, "Return along the recorded path to the original joints")
            planner.playback(trajectory)
            segments.append(trajectory)
        if not np.allclose(planner.scene.data.qpos[planner.scene.qadr], start, atol=1e-9, rtol=0):
            raise PlanningError("Simulated return did not reach the original joints")
        return ExperimentPlan(tuple(start), tuple(upright), tuple(segments), outward_count,
                              tuple(decisions), chooser.model, original_state)
    finally:
        planner.set_start(start)


def preview(planner, plan):
    """One continuous viewer session, including a pause at the upright pose."""
    import mujoco.viewer
    from nero_planner.planner import smoothstep

    planner.set_start(plan.original_joints)
    try:
        with mujoco.viewer.launch_passive(planner.scene.model, planner.scene.data) as window:
            window.opt.geomgroup[3] = 0
            for index, trajectory in enumerate(plan.trajectories):
                planner.prepare_playback(trajectory)
                start, goal = np.asarray(trajectory.positions)[[0, -1]]
                began = time.monotonic()
                duration = trajectory.timestamps[-1]
                while True:
                    if not window.is_running():
                        raise PlanningError("Preview cancelled; hardware execution cancelled")
                    elapsed = min(time.monotonic() - began, duration)
                    with window.lock():
                        planner.scene.set_arm(start + smoothstep(elapsed / duration) * (goal - start))
                    window.sync()
                    if elapsed >= duration:
                        break
                    time.sleep(0.02)
                if index + 1 == plan.outward_count:
                    time.sleep(0.5)
    finally:
        planner.set_start(plan.original_joints)
