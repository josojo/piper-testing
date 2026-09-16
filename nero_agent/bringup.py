"""Start a headless ROS stack and run the CLI; clean up only our processes."""
import argparse
import json
import tempfile
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .core import Settings, strict_json

ROOT = Path(__file__).resolve().parents[1]


def stop_processes(processes):
    """Finish bounded cleanup even if the user presses Ctrl-C again."""
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        for process in reversed(processes):
            if process.poll() is not None:
                continue
            for sig, timeout in ((signal.SIGINT, 8), (signal.SIGTERM, 3), (signal.SIGKILL, 3)):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    break
                try:
                    process.wait(timeout=timeout)
                    break
                except subprocess.TimeoutExpired:
                    continue
    finally:
        signal.signal(signal.SIGINT, previous)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--diagnose-feedback', action='store_true')
    mode.add_argument('--commission-abort', action='store_true')
    mode.add_argument('--capture-only', action='store_true')
    mode.add_argument('--validate-hold', action='store_true')
    args, remaining = parser.parse_known_args()
    settings = Settings.parse(strict_json(args.config.read_text()), require_review=args.commission_abort or '--execute' in remaining)
    if settings.mode == 'hardware' and '--execute' in remaining:
        from .abort_policy import require_verified_controlled_abort
        try:
            qualification = None
            if '--abort-qualification-report' in remaining:
                index = remaining.index('--abort-qualification-report')
                if index + 1 < len(remaining):
                    qualification = remaining[index + 1]
            require_verified_controlled_abort(qualification)
        except Exception as error:
            print(str(error), file=sys.stderr)
            return 2
    diagnostic_args = None
    if args.diagnose_feedback:
        import math
        diagnostic_parser = argparse.ArgumentParser()
        diagnostic_parser.add_argument('--duration', type=float, default=30.)
        diagnostic_parser.add_argument('--output', type=Path, default=ROOT / 'reports/agent-feedback-stack.json')
        diagnostic_args = diagnostic_parser.parse_args(remaining)
        if settings.mode != 'hardware' or not math.isfinite(diagnostic_args.duration) or not 1 <= diagnostic_args.duration <= 120:
            parser.error('Full-stack diagnostic requires hardware config and duration between 1 and 120 seconds')
        if diagnostic_args.output.resolve() == args.config.resolve():
            parser.error('Output must not overwrite configuration')
        from .__main__ import write_report
        write_report(diagnostic_args.output, {'status': 'starting', 'motion_commands_sent': False})
    processes = []
    logs = []
    initial_file = None
    directory = ROOT / 'reports'
    directory.mkdir(exist_ok=True)
    try:
        commands = [
            [sys.executable, '-m', 'nero_agent.driver' if settings.mode == 'hardware' else 'nero_agent.mock_info',
             '--namespace', settings.namespace],
            ['ros2', 'launch', str(ROOT / 'ros2/nero.launch.py'), 'mode:=' + settings.mode,
             'namespace:=' + settings.namespace.strip('/')],
        ]
        if args.diagnose_feedback:
            commands[0].extend(['--diagnose-feedback', '--diagnostic-duration', str(diagnostic_args.duration)])
        if args.commission_abort:
            if settings.mode != 'hardware':
                raise RuntimeError('Commissioning requires hardware configuration')
            commands[0].append('--commission-abort')
        if settings.mode == 'hardware' and '--execute' in remaining and '--abort-qualification-report' in remaining:
            index = remaining.index('--abort-qualification-report')
            if index + 1 < len(remaining):
                commands[0].extend(['--abort-qualification-report', remaining[index + 1]])
        from .ros_backend import RosBackend
        for index, cmd in enumerate(commands):
            if index == 1 and settings.mode == 'hardware':
                deadline = time.monotonic() + 30
                while True:
                    try:
                        backend = RosBackend(settings)
                        try:
                            initial = backend.state()
                        finally:
                            backend.close()
                        break
                    except Exception as error:
                        print('Waiting for hardware feedback: %s' % error, file=sys.stderr, flush=True)
                        if time.monotonic() > deadline or processes[0].poll() is not None:
                            raise
                        time.sleep(0.5)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                    json.dump(initial, f)
                    initial_file = Path(f.name)
                cmd.append('initial_state_file:=' + str(initial_file))
            logfile = (directory / ('ros2-%s-%d.log' % (settings.mode, index))).open('w')
            logs.append(logfile)
            processes.append(subprocess.Popen(cmd, stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True))
        # Readiness checks are read-only; never open the hardware gate here.
        from .ros_backend import RosBackend
        deadline, last_error = time.monotonic() + 90, ''
        print('Starting ROS stack; logs: reports/ros2-%s-*.log' % settings.mode, flush=True)
        while time.monotonic() < deadline:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError('ROS launch process exited; inspect reports/ros2-*.log')
            try:
                backend = RosBackend(settings)
                try:
                    ready = backend.planner.wait_for_service(timeout_sec=2) and backend.executor.wait_for_server(timeout_sec=2)
                finally:
                    backend.close()
                if ready:
                    break
                last_error = 'MoveIt planning service or execution action is not available yet'
            except Exception as error:
                last_error = str(error)
            print('Waiting for ROS readiness: ' + last_error, file=sys.stderr, flush=True)
            time.sleep(0.5)
        else:
            raise RuntimeError('ROS stack did not become ready: ' + last_error + '; inspect reports/ros2-*.log')
        print('ROS 2 + MoveIt ready (%s, gripper modeled, arm-only commands).' % settings.mode, flush=True)
        if args.diagnose_feedback:
            from .feedback_diagnostic import full_stack
            return full_stack(settings, diagnostic_args.output, diagnostic_args.duration, processes)
        from .__main__ import main as cli
        command = 'commission-abort' if args.commission_abort else 'validate-hold' if args.validate_hold else ('capture' if args.capture_only else 'run')
        return cli([command, '--config', str(args.config)] + remaining)
    except KeyboardInterrupt:
        if diagnostic_args:
            write_report(diagnostic_args.output, {'status': 'interrupted', 'motion_commands_sent': False})
        return 130
    except Exception as error:
        if diagnostic_args:
            write_report(diagnostic_args.output, {'status': 'diagnostic_failed', 'reason': str(error), 'motion_commands_sent': False})
        print(str(error), file=sys.stderr)
        return 2
    finally:
        # Stop the controller processes before taking down the hardware bridge.
        stop_processes(processes)
        for log in logs:
            log.close()
        if initial_file is not None:
            initial_file.unlink(missing_ok=True)


if __name__ == '__main__':
    raise SystemExit(main())
