"""Small bounds/watchdog guard for the ROS-to-NERO transport, not a planner."""
import math
from collections import deque
from .core import AgentError, distance, vector, MAX_EXCURSION_RAD, excursion_detail, TRACKING_TOLERANCE_RAD


class StreamGuard:
    MOTOR_MEDIAN_SAMPLES = 5
    MOTOR_MEDIAN_WINDOW_S = .050
    _POSITION_TIMESTAMP_INDEX = (0, 0, 1, 1, 2, 2, 3)

    def __init__(self, start, gripper, now, motor_velocity_limit=0.10,
                 max_excursion=MAX_EXCURSION_RAD, command_velocity_limit=.10,
                 position_velocity_limit=.15):
        self.start = vector(start)
        self.gripper = gripper
        self.motor_velocity_limit = float(motor_velocity_limit)
        self.max_excursion = max_excursion
        self.command_velocity_limit = command_velocity_limit
        self.position_velocity_limit = position_velocity_limit
        self.last_command = None
        self.last_stamp = None
        self.command_at = now
        self.heartbeat_at = now
        self.history = deque(maxlen=400)
        self._last_feedback_joints = None
        self._last_position_timestamps = None
        self._motor_samples = [deque(maxlen=self.MOTOR_MEDIAN_SAMPLES) for _ in range(7)]
        self._motor_stamps = [None] * 7

    def check_feedback(self, joints, velocities, gripper, now,
                       velocity_timestamps=None, wall_now=None,
                       position_timestamps=None):
        q, velocity = vector(joints), vector(velocities)
        event = {'event': 'feedback', 'monotonic_s': now,
                             'joints_rad': q, 'velocities_rad_s': velocity,
                             'last_command_rad': self.last_command}
        if position_timestamps is not None:
            position_stamps = tuple(position_timestamps)
            event['joint_position_timestamps'] = position_stamps
            if len(position_stamps) != 4:
                raise AgentError('Expected four grouped joint-position timestamps')
            if (self._last_feedback_joints is not None and
                    self._last_position_timestamps is not None):
                derived = []
                discrepancy = []
                for i, stamp_index in enumerate(self._POSITION_TIMESTAMP_INDEX):
                    stamp = position_stamps[stamp_index]
                    dt = stamp - self._last_position_timestamps[stamp_index]
                    if dt > 0:
                        measured_velocity = ((q[i] - self._last_feedback_joints[i]) / dt)
                        derived.append(measured_velocity)
                        discrepancy.append(velocity[i] - measured_velocity)
                    else:
                        derived.append(None)
                        discrepancy.append(None)
                event['joint_finite_difference_velocities_rad_s'] = tuple(derived)
                event['motor_minus_position_velocity_rad_s'] = tuple(discrepancy)
            self._last_feedback_joints = q
            self._last_position_timestamps = position_stamps
        if velocity_timestamps is not None:
            event['motor_velocity_timestamps'] = tuple(velocity_timestamps)
            if wall_now is not None:
                event['motor_velocity_ages_s'] = tuple(wall_now - stamp for stamp in velocity_timestamps)
        self.history.append(event)
        # Median absolute speed rejects up to two isolated outliers without allowing
        # direction reversals to cancel. Three fresh high samples suffice, even
        # during startup. Never count repeated SDK cache reads as new evidence.
        filtered = []
        for i, speed in enumerate(velocity):
            samples = self._motor_samples[i]
            while samples and now - samples[0][0] > self.MOTOR_MEDIAN_WINDOW_S:
                samples.popleft()
            stamp = now if velocity_timestamps is None else velocity_timestamps[i]
            if self._motor_stamps[i] is None or stamp > self._motor_stamps[i]:
                samples.append((now, abs(speed)))
                self._motor_stamps[i] = stamp
            values = sorted([value for _, value in samples] + [0.] * (self.MOTOR_MEDIAN_SAMPLES - len(samples)))
            filtered.append(values[self.MOTOR_MEDIAN_SAMPLES // 2])
        event['motor_median_speeds_rad_s'] = tuple(filtered)
        raw_index = max(range(7), key=lambda i: abs(velocity[i]))
        immediate = abs(velocity[raw_index]) > 2 * self.motor_velocity_limit
        index = raw_index if immediate else max(range(7), key=lambda i: filtered[i])
        if immediate or filtered[index] > self.motor_velocity_limit:
            timing = ''
            if 'motor_velocity_ages_s' in event:
                timing = '; motor velocity ages ' + ', '.join(
                    'joint%d=%.1fms' % (i + 1, age * 1000)
                    for i, age in enumerate(event['motor_velocity_ages_s']))
            basis = ('instantaneous hard limit %.3f rad/s' % (2 * self.motor_velocity_limit)
                     if immediate else '%d-sample median speed %.6f rad/s' % (self.MOTOR_MEDIAN_SAMPLES, filtered[index]))
            error = AgentError('Measured velocity exceeded %.3f rad/s: joint%d %.6f rad/s; '
                               'measured %.6f rad, last command %s%s' %
                               (self.motor_velocity_limit, index + 1, velocity[index], q[index],
                                'none' if self.last_command is None else
                                '%.6f rad' % self.last_command[index], timing) + '; ' + basis)
            error.history = list(self.history)
            raise error
        derived = event.get('joint_finite_difference_velocities_rad_s')
        if derived is not None:
            derived_index = max((i for i, value in enumerate(derived) if value is not None),
                                key=lambda i: abs(derived[i]), default=None)
            if derived_index is not None and abs(derived[derived_index]) > self.position_velocity_limit:
                error = AgentError('Position-derived velocity exceeded %.3f rad/s: joint%d %.6f rad/s; '
                                   'measured %.6f rad' %
                                   (self.position_velocity_limit, derived_index + 1, derived[derived_index], q[derived_index]))
                error.history = list(self.history)
                raise error
        if distance(joints, self.start) > self.max_excursion + 0.01:
            raise AgentError('Measured joints left the execution excursion envelope: ' +
                             excursion_detail(joints, self.start, self.max_excursion + 0.01))
        if not math.isfinite(gripper) or abs(gripper - self.gripper) > 0.001:
            raise AgentError('Gripper opening changed during arm execution')
        if now - self.command_at > 0.25 or now - self.heartbeat_at > 0.5:
            raise AgentError('Controller stream or application heartbeat timed out')

    def command(self, joints, measured, stamp, wall_now, now, position_timestamps=None):
        q = vector(joints)
        measured = vector(measured)
        event = {'event': 'command_received', 'monotonic_s': now,
                 'received_unix_s': wall_now, 'stamp': stamp, 'joints_rad': q,
                 'measured_rad': measured, 'position_error_rad': tuple(a-b for a, b in zip(q, measured)),
                 'previous_command_rad': self.last_command, 'previous_command_stamp': self.last_stamp,
                 'tracking_limit_rad': TRACKING_TOLERANCE_RAD,
                 'command_velocity_limit_rad_s': self.command_velocity_limit}
        if position_timestamps is not None:
            event['joint_position_timestamps'] = tuple(position_timestamps)
        self.history.append(event)
        if not math.isfinite(stamp) or not 0 <= wall_now - stamp <= 0.1:
            raise AgentError('Controller command timestamp is stale or invalid')
        if distance(q, self.start) > self.max_excursion + 1e-6:
            raise AgentError('Controller command exceeded excursion bounds: ' +
                             excursion_detail(q, self.start, self.max_excursion))
        if distance(q, measured) > TRACKING_TOLERANCE_RAD:
            raise AgentError('Controller command exceeded excursion/tracking bounds')
        if self.last_command is None:
            if distance(q, measured) > 0.005:
                raise AgentError('Controller initial position does not match measured arm')
        else:
            dt = stamp - self.last_stamp
            delta = distance(q, self.last_command)
            allowance = self.command_velocity_limit * max(dt, 0.01) + 1e-5
            event.update(command_dt_s=dt, max_command_delta_rad=delta, allowed_command_delta_rad=allowance)
            if dt <= 0 or delta > allowance:
                raise AgentError('Controller command stream exceeded velocity limit: '
                                 'dt %.9f s, largest step %.9f rad, allowed %.9f rad; limit %.3f rad/s' %
                                 (dt, delta, allowance, self.command_velocity_limit))
        self.last_command, self.last_stamp, self.command_at = q, stamp, now
        return q
