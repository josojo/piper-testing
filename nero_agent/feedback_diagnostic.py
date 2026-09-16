"""Passive SDK-cache and ROS timer timing; never opens a motion gate."""
import argparse
import math
from pathlib import Path
import time

PACKETS = ('joint_12', 'joint_34', 'joint_56', 'joint_7')


def packet_snapshot(robot, wall):
    parser = getattr(robot, '_parser', None)
    stamps = [getattr(getattr(parser, name, None), 'timestamp', None) for name in PACKETS]
    stamps = [float(s) if isinstance(s, (int, float)) and math.isfinite(s) else None for s in stamps]
    return {'source_stamps': stamps, 'ages_s': [wall-s if s is not None else None for s in stamps]}


class Recorder:
    def __init__(self, hardware, clock=time.monotonic, wall=time.time):
        self.hardware, self.clock, self.wall = hardware, clock, wall
        self.samples = []
        self.previous = None

    def begin(self):
        started = self.clock()
        sample = {'monotonic_s': started, 'wall_s': self.wall(),
                  'timer_interval_s': None if self.previous is None else started-self.previous}
        self.previous = started
        sample['before'] = packet_snapshot(self.hardware.robot, self.wall())
        self.current = sample

    def read(self, require_enabled=False):
        sample = self.current
        read_start = self.clock()
        try:
            state = self.hardware.read(require_enabled=require_enabled)
            sample['joints_rad'] = list(state.joints_rad)
            sample['velocities_rad_s'] = list(state.velocities_rad_s)
            sample['enabled'] = list(state.enabled)
            sample['joint_position_timestamps'] = list(getattr(state, 'joint_position_timestamps', ()))
            sample['motor_velocity_timestamps'] = list(getattr(state, 'motor_velocity_timestamps', ()))
            sample['captured_at_unix'] = getattr(state, 'captured_at_unix', None)
            sample['error'] = None
            return state
        except Exception as error:
            sample['error'] = str(error) or type(error).__name__
            raise
        finally:
            sample['read_duration_s'] = self.clock()-read_start
            sample['after'] = packet_snapshot(self.hardware.robot, self.wall())

    def finish(self):
        self.current['callback_duration_s'] = self.clock()-self.current['monotonic_s']
        self.samples.append(self.current)

    def tick(self):
        self.begin()
        try:
            self.read()
        except Exception:
            pass
        finally:
            self.finish()

    def result(self):
        def stats(values):
            v = sorted(x for x in values if x is not None)
            return {'count': len(v), 'max_s': max(v) if v else None,
                    'p99_s': v[min(len(v)-1, int(len(v)*.99))] if v else None}
        summary = {key: stats(s[key] for s in self.samples) for key in
                   ('timer_interval_s', 'read_duration_s', 'callback_duration_s')}
        summary['bridge_errors'] = sum(bool(s.get('bridge_error')) for s in self.samples)
        summary['read_errors'] = sum(s['error'] is not None for s in self.samples)
        summary['successful_reads'] = len(self.samples)-summary['read_errors']
        summary['packets'] = {}
        for i, name in enumerate(PACKETS):
            stamps = [s['before']['source_stamps'][i] for s in self.samples]
            advances = [b-a for a,b in zip(stamps, stamps[1:]) if a is not None and b is not None and b>a]
            summary['packets'][name] = {'age': stats(s['before']['ages_s'][i] for s in self.samples),
                                        'observed_source_advance': stats(advances)}
        return {'status': 'diagnostic_completed', 'summary': summary, 'samples': self.samples,
                'motion_commands_sent': False,
                'scope': 'Standalone ROS 100 Hz timer and existing checked SDK reader. Cache polling, not raw CAN capture; source advances can span missed packets. No MoveIt/controller load or active holding verification.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('reports/agent-feedback-diagnostic.json'))
    parser.add_argument('--duration', type=float, default=30.)
    args = parser.parse_args(argv)
    from .core import Settings, strict_json
    from .__main__ import write_report
    if not math.isfinite(args.duration) or not 1 <= args.duration <= 120:
        parser.error('duration must be between 1 and 120 seconds')
    if args.output.resolve() == args.config.resolve():
        parser.error('output must not overwrite configuration')
    settings = Settings.parse(strict_json(args.config.read_text()), require_review=False)
    if settings.mode != 'hardware':
        parser.error('diagnostic requires hardware configuration')
    # Check report destination before opening the SDK connection.
    write_report(args.output, {'status': 'starting', 'motion_commands_sent': False})
    import rclpy
    from nero_experiment.hardware import connect
    recorder = None
    node = None
    interrupted = False
    failure = None
    rclpy.init()
    try:
        with connect() as hardware:
            recorder = Recorder(hardware)
            node = rclpy.create_node('nero_feedback_diagnostic', namespace=settings.namespace)
            node.create_timer(.01, recorder.tick)
            print('Passive feedback diagnostic for %.0f seconds; no motion commands. Keep other controllers stopped.' % args.duration, flush=True)
            end = time.monotonic()+args.duration
            while time.monotonic() < end:
                rclpy.spin_once(node, timeout_sec=.05)
    except KeyboardInterrupt:
        interrupted = True
    except Exception as error:
        failure = str(error) or type(error).__name__
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if recorder is not None:
            report = recorder.result()
            if interrupted:
                report['status'] = 'interrupted'
            if failure:
                report.update(status='diagnostic_failed', reason=failure)
            write_report(args.output, report)
    if recorder is not None:
        import json
        print(json.dumps({'status': report['status'], 'report': str(args.output), 'summary': report['summary']}))
    if recorder is None and failure:
        write_report(args.output, {'status': 'diagnostic_failed', 'reason': failure, 'motion_commands_sent': False})
        print(failure)
    return 130 if interrupted else 2 if failure else 0


if __name__ == '__main__':
    raise SystemExit(main())


def full_stack(settings, output, duration, processes):
    """Collect inside the existing bridge; never open a second SDK connection."""
    import json
    from std_srvs.srv import Trigger
    from .ros_backend import RosBackend
    from .__main__ import write_report
    backend = RosBackend(settings, require_ready=False)
    try:
        start = backend.node.create_client(Trigger, settings.namespace + '/project/diagnostic_start')
        result = backend.node.create_client(Trigger, settings.namespace + '/project/diagnostic_result')
        response = backend._call(start, Trigger.Request())
        if not response.success:
            raise RuntimeError(response.message)
        print('Recording stationary feedback inside the full ROS stack for %.0f seconds; hardware writes blocked.' % duration, flush=True)
        status_requests = 0
        largest_status_bytes = 0
        deadline = time.monotonic()+duration+2
        while time.monotonic() < deadline:
            if any(p.poll() is not None for p in processes):
                raise RuntimeError('ROS process exited during diagnostic')
            status = backend.poll_abort_status()
            payload = json.loads(status.message)
            if not status.success or payload['status'] != 'idle':
                raise RuntimeError('Unexpected abort state during stationary diagnostic')
            if 'history' in payload.get('commissioning', {}):
                raise RuntimeError('Live status unexpectedly contains full trace')
            status_requests += 1
            largest_status_bytes = max(largest_status_bytes, len(status.message.encode()))
        response = backend._call(result, Trigger.Request())
        if not response.success:
            raise RuntimeError(response.message)
        report = json.loads(response.message)
        report['reporting_test'] = {'status_requests': status_requests,
                                    'max_live_response_bytes': largest_status_bytes,
                                    'poll_interval_s': .1,
                                    'full_trace': backend.fetch_abort_report({'status': 'idle'})}
        report['summary']['reporting_status_requests'] = status_requests
        report['summary']['max_live_response_bytes'] = largest_status_bytes
        write_report(output, report)
        print(json.dumps({'status': report['status'], 'report': str(output), 'summary': report['summary']}))
        return 0
    finally:
        backend.close()
