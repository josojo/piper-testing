"""Offline abort trace analysis. Never connects to ROS or hardware."""
import argparse
import json
from pathlib import Path


def analyze(report):
    abort = report['controlled_abort']
    history = abort.get('commissioning', {}).get('history', [])
    reference = abort.get('timings', {}).get('move_j_returned_monotonic_s')
    if not history or reference is None:
        raise ValueError('Report needs feedback history and a completed hold-command timestamp')
    rows = []
    for joint in range(7):
        post = [s for s in history if s['monotonic_s'] >= reference]
        if not post:
            raise ValueError('No feedback after hold command')
        peak = max(post, key=lambda s: abs(s['velocities_rad_s'][joint]))
        index = history.index(peak)
        neighbors = history[max(0, index-2):index+3]
        position_stamps = peak.get('joint_position_timestamps', [])
        motor_stamps = peak.get('motor_velocity_timestamps', [])
        paired = len(position_stamps) == 4 and len(motor_stamps) == 7
        rows.append({
            'joint': joint+1,
            'peak_reported_velocity_rad_s': peak['velocities_rad_s'][joint],
            'peak_time_after_hold_s': peak['monotonic_s']-reference,
            'max_position_error_from_hold_rad': max(
                abs(s['joints_rad'][joint]-abort['hold_target_rad'][joint]) for s in post),
            'motor_minus_position_timestamp_s': (
                motor_stamps[joint]-position_stamps[joint//2] if paired else None),
            'peak_neighborhood': [{
                'time_after_hold_s': s['monotonic_s']-reference,
                'position_rad': s['joints_rad'][joint],
                'reported_velocity_rad_s': s['velocities_rad_s'][joint],
            } for s in neighbors],
        })
    return {'scope': 'Offline observations only; does not validate safety or unlock motion.',
            'timing_note': 'Observation time is not packet acquisition time. Missing packet timestamps prevent exact position/velocity alignment. SDK timestamps are not raw CAN capture.',
            'abort_reason': abort.get('reason'), 'failure': abort.get('failure'),
            'deliberate_trigger_observed': abort.get('commissioning', {}).get('trigger_state') is not None,
            'joints': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.report.resolve() == args.output.resolve():
        parser.error('output must differ from the input report')
    result = analyze(json.loads(args.report.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({'status': 'analyzed', 'output': str(args.output)}))


if __name__ == '__main__':
    main()
