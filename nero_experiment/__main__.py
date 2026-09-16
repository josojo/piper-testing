"""CLI for state capture, OpenRouter planning, simulation and opt-in execution."""

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import sys

from nero_planner import Planner, PlanningError
from .hardware import HardwareConfig, connect, execute, hardware_schedule, check_calibration
from .openrouter import DEFAULT_MODEL, OfflineChooser, OpenRouter, strict_json
from .workflow import DEMO_START, EXPERIMENT_LIMITS, UPRIGHT, plan_experiment, preview


ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return strict_json(path.read_text())


def write_report(path, report, encoded_trajectories=None):
    # Never leave a partially overwritten report if execution is interrupted.
    temporary = path.with_name(path.name + ".tmp")
    if encoded_trajectories is None:
        text = json.dumps(report, indent=2, allow_nan=False)
    else:
        # Trajectories are immutable during execution. Re-encoding their large
        # arrays at every microstep can starve the Python CAN receiver thread.
        dynamic = {key: value for key, value in report.items() if key != "trajectories"}
        text = json.dumps(dynamic, allow_nan=False)[:-1] + ', "trajectories": ' + encoded_trajectories + '}'
    temporary.write_text(text + "\n")
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    capture = commands.add_parser("capture", help="Read stationary hardware state without enabling or moving")
    capture.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Simulate the complete experiment; optionally execute on hardware")
    source = run.add_mutually_exclusive_group()
    source.add_argument("--start", type=Path, help="Seven-radian array or saved capture JSON")
    source.add_argument("--read-hardware", action="store_true", help="Capture current hardware feedback")
    run.add_argument("--scene", type=Path, default=ROOT / "models/nero/nero_scene.xml")
    run.add_argument("--upright", type=Path, help="Simulation-only upright joint reference array; default all zero")
    run.add_argument("--model", default=DEFAULT_MODEL)
    run.add_argument("--offline-demo", action="store_true", help="Use an explicit LLM test double; cannot execute")
    run.add_argument("--viewer", action="store_true")
    run.add_argument("--execute", action="store_true", help="Enable supervised physical execution after simulation")
    run.add_argument("--hardware-config", type=Path, help="Reviewed physical setup and flange calibration")
    run.add_argument("--max-steps", type=int, default=3, help="Maximum outward pose requests (1-64); at most 3 LLM attempts per request")
    run.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = {"status": "starting", "hardware_executed": False}
    events = []
    report_writable = False

    def progress(message):
        print(message, file=sys.stderr, flush=True)
        events.append(message)

    try:
        for name in ("start", "scene", "upright", "hardware_config"):
            source_path = getattr(args, name, None)
            if source_path is not None and source_path.resolve() == args.output.resolve():
                raise PlanningError("Output report must not overwrite an input file")
        # Fail on unwritable output before connecting, billing or motion.
        write_report(args.output, report)
        report_writable = True
        if args.command == "capture":
            with connect() as hardware:
                state = hardware.initial_state()
            write_report(args.output, state.to_dict())
            print(json.dumps({"status": "captured", "output": str(args.output), "joints_rad": state.joints_rad}))
            return 0
        if not 1 <= args.max_steps <= 64:
            raise PlanningError("max-steps must be between 1 and 64")
        if args.execute and (not args.read_hardware or args.offline_demo or args.hardware_config is None):
            raise PlanningError("--execute requires --read-hardware and --hardware-config, and forbids --offline-demo")
        if args.upright and args.hardware_config:
            raise PlanningError("Use the upright reference in hardware-config, not --upright, for hardware runs")
        if not args.read_hardware and args.start is None and not args.offline_demo:
            raise PlanningError("Supply --start or --read-hardware (or --offline-demo for the built-in simulated start)")
        config = HardwareConfig.from_dict(read_json(args.hardware_config)) if args.hardware_config else None
        upright = config.upright_joints_rad if config else (read_json(args.upright) if args.upright else UPRIGHT)
        planner = Planner(args.scene, EXPERIMENT_LIMITS)
        chooser = OfflineChooser() if args.offline_demo else OpenRouter(args.model)
        with connect() if args.read_hardware else nullcontext(None) as hardware:
            if hardware is not None:
                state = hardware.initial_state()
                start, observation = state.joints_rad, state.to_dict()
                if config:
                    check_calibration(planner, state, config)
            else:
                value = read_json(args.start) if args.start else list(DEMO_START)
                if isinstance(value, dict):
                    start = value["joints_rad"]
                    observation = {**value, "source": "recorded_state_file_not_live_feedback"}
                else:
                    start, observation = value, {"source": "simulation", "joints_rad": value}
            plan = plan_experiment(planner, start, chooser, upright, observation, args.max_steps, progress)
            report = {**plan.to_dict(), "scene_fingerprint": planner.scene.fingerprint(), "events": events}
            if config:
                schedule = hardware_schedule(planner, plan)
                report["hardware_envelope_preflight"] = {"status": "passed", "microsteps": len(schedule)}
            write_report(args.output, report)
            progress(f"Complete simulation passed: {plan.outward_count} outward and {len(plan.trajectories) - plan.outward_count} return segments")
            if not plan.trajectories:
                progress("Captured arm is already at the upright reference; no motion needed")
            if args.viewer:
                preview(planner, plan)
            if args.execute:
                encoded_trajectories = json.dumps(report["trajectories"], allow_nan=False)
                # Persist state before any hardware command; progress is saved
                # after each settled step and on all handled failures.
                report["status"] = "awaiting_hardware_execution"
                report["hardware_events"] = []
                write_report(args.output, report)

                def hardware_progress(message):
                    progress(message)
                    report["status"] = "hardware_execution_in_progress"
                    write_report(args.output, report, encoded_trajectories)

                def record(event):
                    report["hardware_events"].append(event)
                    write_report(args.output, report, encoded_trajectories)

                final = execute(planner, plan, hardware, config,
                                lambda prompt: input(prompt).strip() == "EXECUTE", hardware_progress, record)
                report.update(status="returned_to_original", hardware_executed=bool(plan.trajectories),
                              final_hardware_state=final.to_dict())
                progress("Returned to original measured joint position. Motor enable state is retained.")
            write_report(args.output, report)
        print(json.dumps({"status": report["status"], "hardware_executed": report["hardware_executed"],
                          "outward_segments": plan.outward_count, "report": str(args.output)}))
        return 0
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="rejected_or_aborted", reason=str(error) or type(error).__name__, events=events)
        # After a failure, "hardware_executed: false" does not mean no motion
        # occurred. Preserve that uncertainty explicitly in the report.
        report["hardware_motion_may_have_occurred"] = bool(getattr(args, "execute", False))
        try:
            if report_writable:
                write_report(args.output, report)
        except OSError:
            pass
        print(json.dumps({"status": report["status"], "reason": report["reason"]}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
