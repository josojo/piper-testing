"""Fixed, single-use moving-abort commissioning bounds and verdicts."""
from collections import deque
import time
from .core import AgentError, distance, vector

CRITERIA = {
    'joint1_goal_delta_rad': 0.01, 'planned_velocity_rad_s': 0.02,
    'planned_acceleration_rad_s2': 0.05, 'moving_displacement_rad': 0.0005,
    'moving_velocity_rad_s': 0.005, 'moving_samples': 2,
    'trigger_deadline_s': 4.0, 'measured_velocity_limit_rad_s': 0.03,
    'max_excursion_rad': 0.012, 'other_joint_excursion_rad': 0.002,
    'stop_time_limit_s': 1.0, 'post_abort_travel_limit_rad': 0.01,
    'stopped_velocity_rad_s': 0.003, 'hold_dwell_s': 2.0,
}


class MovingAbortTrial:
    def __init__(self, start, now):
        self.start = vector(start)
        self.started_at = now
        self.outcome, self.reason = 'pending', None
        self.trigger = None
        self.consecutive = 0
        self.last_source = None
        self.last_command = None
        self.history = deque(maxlen=1000)

    def command(self, joints, stamp, monotonic_s=None):
        q = vector(joints)
        if distance(q, self.start) > .012 or max(abs(q[i] - self.start[i]) for i in range(1, 7)) > .002:
            raise AgentError('Commissioning command leaves fixed joint envelope')
        if self.last_command:
            previous, previous_stamp, _ = self.last_command
            dt = stamp - previous_stamp
            if dt <= 0 or distance(q, previous) > .02 * dt + 0.00002:
                raise AgentError('Commissioning command exceeds 0.02 rad/s bound')
        elif distance(q, self.start) > .0005:
            raise AgentError('Commissioning initial command differs from captured state')
        self.last_command = (q, stamp, monotonic_s if monotonic_s is not None else time.monotonic())

    def observe(self, state, now):
        q, v = vector(state.joints_rad), vector(state.velocities_rad_s)
        stamp = min(state.feedback_timestamps)
        if self.last_source is not None and stamp <= self.last_source:
            if self.outcome == 'pending' and now - self.started_at >= 4:
                self.outcome, self.reason = 'inconclusive', 'No advancing movement feedback before deadline'
                return self.reason
            return None
        self.last_source = stamp
        sample = {'monotonic_s': now, 'source_stamp': stamp,
                  'joints_rad': list(q), 'velocities_rad_s': list(v), 'enabled': list(state.enabled)}
        # Preserve timestamps from the same checked copies as the values; do
        # not re-read SDK caches for diagnostic metadata.
        sample['joint_position_timestamps'] = list(getattr(state, 'joint_position_timestamps', ()))
        sample['motor_velocity_timestamps'] = list(getattr(state, 'motor_velocity_timestamps', ()))
        sample['captured_at_unix'] = getattr(state, 'captured_at_unix', None)
        all_stamps = list(getattr(state, 'feedback_timestamps', ()))
        sample['feedback_packet_age_s'] = [time.time() - stamp for stamp in all_stamps]
        sample['feedback_packet_skew_s'] = (max(all_stamps) - min(all_stamps)) if all_stamps else None
        sample['motion_status'] = getattr(state, 'motion_status', None)
        sample['last_validated_stream_command'] = (
            {'joints_rad': list(self.last_command[0]), 'ros_header_stamp_s': self.last_command[1],
             'monotonic_s': self.last_command[2]}
            if self.last_command else None)
        self.history.append(sample)
        if self.outcome != 'pending':
            return None
        if (max(map(abs, v)) > .03 or distance(q, self.start) > .012 or
                max(abs(q[i] - self.start[i]) for i in range(1, 7)) > .002):
            self.outcome, self.reason = 'failed', 'Commissioning measured motion exceeded its bounds'
        elif q[0] - self.start[0] >= .008:
            self.outcome, self.reason = 'inconclusive', 'Motion reached the goal region before moving-abort trigger'
        elif now - self.started_at >= 4:
            self.outcome, self.reason = 'inconclusive', 'No qualifying movement before deadline'
        else:
            moving = q[0] - self.start[0] >= .0005 and v[0] >= .005
            self.consecutive = self.consecutive + 1 if moving else 0
            if self.consecutive < 2:
                return None
            self.trigger = sample
            self.outcome, self.reason = 'triggered', 'Deliberate abort during measured joint1 motion'
        return self.reason

    def live_status(self):
        # Constant-size response: never scan or serialize the trace in a live callback.
        return {'status': self.outcome, 'reason': self.reason,
                'feedback_samples': len(self.history), 'normal_execution_unlocked': False}

    def result(self, abort, fault=None):
        status = self.outcome
        metrics = {}
        violations = []
        if self.trigger:
            samples = [s for s in self.history if s['monotonic_s'] >= self.trigger['monotonic_s']]
            travel = max((distance(s['joints_rad'], self.trigger['joints_rad']) for s in samples), default=0.)
            peak = max((max(map(abs, s['velocities_rad_s'])) for s in samples), default=0.)
            stable = abort.get('stable_since_monotonic_s')
            first_stable = abort.get('first_stable_since_monotonic_s')
            # Backward-compatible reports predate the separate first-stable
            # marker; those reports used the only available stable timestamp.
            if first_stable is None:
                first_stable = stable
            elapsed = None if stable is None else stable - self.trigger['monotonic_s']
            first_elapsed = None if first_stable is None else first_stable - self.trigger['monotonic_s']
            metrics = {'additional_travel_rad': travel, 'peak_post_abort_velocity_rad_s': peak,
                       'time_to_sustained_standstill_s': elapsed}
            for i in range(7):
                speed = max(abs(s['velocities_rad_s'][i]) for s in samples)
                excursion = max(abs(s['joints_rad'][i] - self.start[i]) for s in samples)
                if speed > .03:
                    violations.append('joint%d post-abort speed %.6f rad/s exceeds 0.030000' % (i+1, speed))
                if i > 0 and excursion > .002:
                    violations.append('joint%d excursion %.6f rad exceeds 0.002000' % (i+1, excursion))
            metrics['standstill_dwell_confirmed'] = abort['status'] == 'holding'
            metrics['time_to_first_standstill_s'] = first_elapsed
            metrics['time_to_sustained_standstill_s'] = elapsed
            metrics['max_observation_gap_s'] = max(
                (b['monotonic_s'] - a['monotonic_s'] for a, b in zip(samples, samples[1:])), default=0.)
            ages = [age for s in samples for age in s.get('feedback_packet_age_s', [])]
            skews = [s['feedback_packet_skew_s'] for s in samples
                     if s.get('feedback_packet_skew_s') is not None]
            commands = [s['last_validated_stream_command']['monotonic_s'] for s in samples
                        if s.get('last_validated_stream_command')]
            metrics['max_feedback_packet_age_s'] = max(ages, default=None)
            metrics['max_feedback_packet_skew_s'] = max(skews, default=None)
            metrics['max_controller_command_gap_s'] = max(
                (b-a for a, b in zip(commands, commands[1:])), default=None)
            if abort['status'] == 'holding':
                status = ('passed' if first_elapsed is not None and 0 <= first_elapsed <= 1.0 and travel <= .01
                          and peak <= .03 and abort['controller_cancellation'] == 'acknowledged'
                          and all(all(s['enabled']) for s in samples)
                          and all(distance(s['joints_rad'], self.start) <= .012
                                  and max(abs(s['joints_rad'][i] - self.start[i]) for i in range(1, 7)) <= .002
                                  for s in samples) else 'failed')
        if fault or abort['status'] == 'failed':
            status = 'failed'
        return {'status': status, 'reason': fault or abort.get('failure') or self.reason,
                'observed_violations': violations,
                'criteria': CRITERIA, 'trigger_state': self.trigger, 'metrics': metrics,
                'history': list(self.history), 'normal_execution_unlocked': False}


def validate_trial_plan(trajectory, start):
    from dataclasses import replace
    from .core import Settings
    from .trajectory import validate_trajectory
    start = vector(start)
    goal = list(start)
    goal[0] += .01
    settings = replace(Settings('hardware', {}, ()), max_excursion=.012, max_velocity=.02,
                       max_acceleration=.05, tolerance=.000501, timeout=4.)
    validate_trajectory(trajectory, start, goal, start, settings)
    for point in trajectory.points:
        q = vector(point.positions)
        if not -.0005 <= q[0] - start[0] <= .0105 or max(abs(q[i] - start[i]) for i in range(1, 7)) > .002:
            raise AgentError('Commissioning plan leaves fixed joint1 test envelope')
