"""Small bounds/watchdog guard for the ROS-to-NERO transport, not a planner."""
import math
from collections import deque
from .core import AgentError, distance, vector


class StreamGuard:
    def __init__(self, start, gripper, now):
        self.start = vector(start)
        self.gripper = gripper
        self.last_command = None
        self.last_stamp = None
        self.command_at = now
        self.heartbeat_at = now
        self.history = deque(maxlen=60)

    def check_feedback(self, joints, velocities, gripper, now):
        q, velocity = vector(joints), vector(velocities)
        self.history.append({'event': 'feedback', 'monotonic_s': now,
                             'joints_rad': q, 'velocities_rad_s': velocity,
                             'last_command_rad': self.last_command})
        index = max(range(7), key=lambda i: abs(velocity[i]))
        if abs(velocity[index]) > 0.10:
            error = AgentError('Measured velocity exceeded 0.10 rad/s: joint%d %.6f rad/s; '
                               'measured %.6f rad, last command %s' %
                               (index + 1, velocity[index], q[index],
                                'none' if self.last_command is None else
                                '%.6f rad' % self.last_command[index]))
            error.history = list(self.history)
            raise error
        if distance(joints, self.start) > 0.16:
            raise AgentError('Measured joints left the execution excursion envelope')
        if not math.isfinite(gripper) or abs(gripper - self.gripper) > 0.001:
            raise AgentError('Gripper opening changed during arm execution')
        if now - self.command_at > 0.25 or now - self.heartbeat_at > 0.5:
            raise AgentError('Controller stream or application heartbeat timed out')

    def command(self, joints, measured, stamp, wall_now, now):
        q = vector(joints)
        self.history.append({'event': 'command_received', 'monotonic_s': now,
                             'stamp': stamp, 'joints_rad': q,
                             'measured_rad': vector(measured)})
        if not math.isfinite(stamp) or not 0 <= wall_now - stamp <= 0.1:
            raise AgentError('Controller command timestamp is stale or invalid')
        if distance(q, self.start) > 0.15 + 1e-6 or distance(q, measured) > 0.015:
            raise AgentError('Controller command exceeded excursion/tracking bounds')
        if self.last_command is None:
            if distance(q, measured) > 0.005:
                raise AgentError('Controller initial position does not match measured arm')
        else:
            dt = stamp - self.last_stamp
            if dt <= 0 or distance(q, self.last_command) > 0.10 * max(dt, 0.01) + 1e-5:
                raise AgentError('Controller command stream exceeded velocity limit')
        self.last_command, self.last_stamp, self.command_at = q, stamp, now
        return q
