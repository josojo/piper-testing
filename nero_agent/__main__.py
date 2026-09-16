"""Simple CLI for offline plumbing, MoveIt tests and supervised NERO actions."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

from .core import AgentError, OfflineBackend, ScriptedChooser, Settings, run_loop, strict_json

ROOT = Path(__file__).resolve().parents[1]


def write_report(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor', help='Check local ROS Python imports without connecting to hardware')
    for name in ('run', 'capture', 'stop', 'validate-hold', 'commission-abort'):
        p = sub.add_parser(name)
        p.add_argument('--config', type=Path, default=ROOT / 'examples/nero-agent.mock.json')
        p.add_argument('--output', type=Path, default=ROOT / ('reports/agent-' + name + '.json'))
        if name == 'run':
            p.add_argument('--instruction', default='Move to inspection, then return to start.')
            choice = p.add_mutually_exclusive_group()
            choice.add_argument('--offline-demo', action='store_true', help='No ROS, LLM, collision checking or hardware')
            choice.add_argument('--scripted', action='store_true', help='Fixed inspection/start sequence instead of LLM')
            p.add_argument('--model')
            p.add_argument('--execute', action='store_true', help='Execute plans; hardware requires per-action confirmation')
            p.add_argument('--abort-qualification-report', type=Path,
                           help='Passed moving-abort report required for hardware execution')
            p.add_argument('--max-actions', type=int, default=8)
    args = parser.parse_args(argv)
    if args.command == 'doctor':
        missing = [m for m in ('rclpy', 'moveit_msgs', 'control_msgs', 'sensor_msgs') if importlib.util.find_spec(m) is None]
        print(json.dumps({'status': 'ready' if not missing else 'missing_ros', 'missing': missing,
                          'hint': 'Use scripts/nero_ros2.sh build, then scripts/nero_ros2.sh demo --scripted'}))
        return 0 if not missing else 2
    backend = None
    report = {'status': 'starting', 'hardware_motion_may_have_occurred': False, 'events': []}
    writable = False
    try:
        if args.output.resolve() == args.config.resolve():
            raise AgentError('Output must not overwrite configuration')
        settings = Settings.parse(strict_json(args.config.read_text()),
                                  require_review=args.command == 'commission-abort' or getattr(args, 'execute', False))
        if args.command == 'commission-abort':
            from dataclasses import replace
            if settings.mode != 'hardware':
                raise AgentError('commission-abort requires hardware configuration')
            settings = replace(settings, max_velocity=.02, max_acceleration=.05, max_excursion=.012, timeout=4., tolerance=.000501)
        offline = getattr(args, 'offline_demo', False)
        if offline and (settings.mode != 'mock' or args.execute):
            raise AgentError('--offline-demo forbids --execute and hardware configuration')
        if settings.mode == 'hardware' and getattr(args, 'execute', False):
            from .abort_policy import require_verified_controlled_abort
            require_verified_controlled_abort(args.abort_qualification_report)
        if args.command == 'validate-hold':
            if settings.mode != 'hardware':
                raise AgentError('validate-hold requires hardware configuration')
            if input('Experimental MOVE J hold test: mechanically support the arm, clear the workspace, '
                     'and stop other controllers. This sends a position command and can move the arm. '
                     'Type SUPPORTED HOLD to proceed: ').strip() != 'SUPPORTED HOLD':
                raise AgentError('Hold validation cancelled; no hold command sent')
        if getattr(args, 'max_actions', 8) not in range(1, 33):
            raise AgentError('max-actions must be between 1 and 32')
        report.update(mode='offline' if offline else settings.mode,
                      backend='offline_test_double' if offline else 'moveit2',
                      instruction=getattr(args, 'instruction', None))
        write_report(args.output, report)
        writable = True
        if offline:
            backend = OfflineBackend(settings)
        else:
            from .ros_backend import RosBackend
            backend = RosBackend(settings, require_ready=args.command != 'stop',
                                 qualification_report=getattr(args, 'abort_qualification_report', None))
        def record(event):
            if event['event'] == 'execution_pending' and backend.hardware:
                report['hardware_motion_may_have_occurred'] = True
            report['events'].append(event)
            write_report(args.output, report)
            if event['event'] in ('decision', 'planned', 'action_result'):
                print('%s: %s' % (event['event'], event.get('pose') or event.get('action') or event.get('status')), file=sys.stderr)
        if args.command == 'commission-abort':
            from .commissioning import CRITERIA
            before = backend.state()
            if max(map(abs, before['velocities_rad_s'])) > .003:
                raise AgentError('Commissioning requires a stationary arm')
            goal = list(before['joints_rad'])
            goal[0] += .01
            plan = backend.plan(goal)
            report.update(initial_state=before, plan=plan, criteria=CRITERIA,
                          validation_scope='Single supported moving-abort observation; normal execution remains blocked')
            write_report(args.output, report)
            if input('Moving-abort test: joint1 target +0.01 rad, planned speed <=0.02 rad/s. '
                     'Mechanically support the arm without obstructing this motion; clear the workspace. '
                     'No automatic return. Type SUPPORTED ABORT to move: ').strip() != 'SUPPORTED ABORT':
                raise AgentError('Moving-abort test cancelled')
            report['hardware_motion_may_have_occurred'] = True
            write_report(args.output, report)
            report.update(backend.commission_abort(plan))
        elif args.command == 'capture':
            report.update(status='captured', state=backend.state())
        elif args.command in ('stop', 'validate-hold'):
            if args.command == 'validate-hold':
                before = backend.state()
                report['initial_state'] = before
                if max(map(abs, before['velocities_rad_s'])) > 0.01:
                    raise AgentError('Supported hold validation must start with a stationary arm')
            report['hardware_motion_may_have_occurred'] = backend.hardware
            write_report(args.output, report)
            report.update(backend.stop())
            if args.command == 'validate-hold':
                report['validation_scope'] = 'Position hold only; not a moving-trajectory abort qualification'
        else:
            if offline or args.scripted:
                chooser = ScriptedChooser()
                if 'inspection' not in settings.named_poses:
                    raise AgentError('The scripted smoke test requires a pose named inspection')
            else:
                from .chooser import OpenRouterChooser
                chooser = OpenRouterChooser(args.model)
            report['model'] = chooser.model
            report.update(run_loop(backend, chooser, settings, args.instruction,
                                   args.execute or offline, record,
                                   lambda prompt: input(prompt).strip() == 'EXECUTE', args.max_actions))
            if offline:
                report['status'] = 'offline_demo_passed'
        write_report(args.output, report)
        print(json.dumps({'status': report['status'], 'mode': report['mode'], 'report': str(args.output)}))
        return 2 if args.command == 'commission-abort' and report['status'] != 'passed' else 0
    except (Exception, KeyboardInterrupt) as error:
        report.update(status='rejected_or_aborted', reason=str(error) or type(error).__name__)
        if backend is not None:
            if getattr(backend, 'last_stop_result', None) is not None:
                report['controlled_abort'] = backend.last_stop_result
            try:
                # Do not send an unsolicited hardware stop for a read-only planning error.
                if getattr(backend, 'motion_pending', False):
                    report['stop_result'] = backend.stop()
            except Exception as stop_error:
                report['stop_error'] = str(stop_error)
        if writable:
            try:
                write_report(args.output, report)
            except OSError:
                pass
        print(json.dumps({'status': report['status'], 'reason': report['reason'],
                          'stop_error': report.get('stop_error')}), file=sys.stderr)
        return 2
    finally:
        if backend is not None:
            backend.close()


if __name__ == '__main__':
    raise SystemExit(main())
