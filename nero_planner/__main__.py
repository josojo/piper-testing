"""CLI for simulation-only target execution; never connects to CAN."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .planner import Limits, Planner, PlanningError, Pose


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, default=ROOT / "models/nero/nero_scene.xml")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--target", type=Path, help="Target JSON file")
    source.add_argument("--demo", action="store_true", help="Short reach from the selected starting pose")
    parser.add_argument("--start", type=Path, help="JSON array of seven joint angles in radians; default: all zero")
    parser.add_argument("--limits", type=Path, help="JSON object overriding Limits defaults")
    parser.add_argument("--output", type=Path, help="Write the validated trajectory and achieved pose as JSON")
    parser.add_argument("--viewer", action="store_true", help="Play back kinematically in the MuJoCo viewer")
    args = parser.parse_args()
    try:
        limits = Limits(**json.loads(args.limits.read_text())) if args.limits else Limits()
        planner = Planner(args.scene, limits)
        planner.set_start(json.loads(args.start.read_text()) if args.start else [0.0] * 7)
        if args.demo:
            initial = planner.current_pose()
            position = list(initial.position_m)
            position[0] += 0.02
            position[2] -= 0.002
            target = Pose("nero_base", position, initial.orientation_xyzw, reason="Short collision-free reach above the table")
        else:
            target = Pose.from_dict(json.loads(args.target.read_text()))
        trajectory = planner.plan(target)
        planner.playback(trajectory, viewer=args.viewer)
        achieved = planner.current_pose()
        # Closing the viewer early is cancellation, not successful execution.
        import numpy as np
        ep, er = planner._errors(planner.scene.data.qpos[planner.scene.qadr], target)
        if np.linalg.norm(ep) > limits.position_tolerance_m or np.linalg.norm(er) > limits.orientation_tolerance_rad:
            raise PlanningError("Playback ended before the target was reached")
        result = {"status": "reached", "execution": "kinematic_simulation",
                  "trajectory": trajectory.to_dict(), "achieved_pose": asdict(achieved)}
        if args.output:
            args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        print(json.dumps({"status": "reached", "execution": "kinematic_simulation",
                          "duration_s": trajectory.timestamps[-1],
                          "validation": asdict(trajectory.validation), "achieved_pose": asdict(achieved)},
                         indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        print(json.dumps({"status": "rejected", "reason": str(error)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
