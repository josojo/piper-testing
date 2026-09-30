"""Bounded controller-state capture and explanations for rejected commands."""
from collections import deque
import math
import time

from .core import JOINTS, TRACKING_TOLERANCE_RAD, vector


class ControllerTrace:
    def __init__(self, capacity=400):
        self.samples = deque(maxlen=capacity)
        self.invalid_samples = 0

    def observe(self, message, received_monotonic=None, received_unix=None):
        try:
            names = list(message.joint_names)
            if len(names) != len(set(names)) or set(names) != set(JOINTS):
                raise ValueError('Unexpected controller joints')
            order = [names.index(name) for name in JOINTS]
            stamp = message.header.stamp.sec + message.header.stamp.nanosec*1e-9
            if not math.isfinite(stamp):
                raise ValueError('Nonfinite timestamp')
            row = {'controller_stamp_unix_s': stamp,
                   'received_monotonic_s': time.monotonic() if received_monotonic is None else received_monotonic,
                   'received_unix_s': time.time() if received_unix is None else received_unix}
            for name in ('desired', 'actual', 'error'):
                point = getattr(message, name)
                positions = vector(point.positions)
                row[name + '_positions_rad'] = [positions[i] for i in order]
                values = vector(point.velocities) if len(point.velocities) else None
                row[name + '_velocities_rad_s'] = [values[i] for i in order] if values else None
            self.samples.append(row)
        except (AttributeError, ValueError, TypeError, RuntimeError, IndexError):
            # Diagnostic failures must never change control behavior.
            self.invalid_samples += 1

    def snapshot(self):
        return {'joint_names': list(JOINTS), 'samples': list(self.samples),
                'invalid_samples': self.invalid_samples,
                'note': 'Controller stamp is not the original hardware measurement timestamp; '
                        'use bridge grouped position timestamps for timing comparisons.'}

    def hold_transition_reason(self, command, stamp):
        """Explain an already rejected command; never authorize a hold jump."""
        if not self.samples:
            return None
        row = self.samples[-1]
        if not 0 <= stamp - row['controller_stamp_unix_s'] <= .05:
            return None
        q = vector(command)
        if max(abs(a-b) for a, b in zip(q, row['actual_positions_rad'])) > 1e-6:
            return None
        errors = row['error_positions_rad']
        index = max(range(len(JOINTS)), key=lambda i: abs(errors[i]))
        if abs(errors[index]) <= TRACKING_TOLERANCE_RAD:
            return None
        return ('Controller tracking tolerance exceeded: %s error %.6f rad, limit %.6f rad; '
                'rejected command matches the controller measured hold position. '
                'Blocking the hold jump and requesting a fresh measured controlled hold'
                % (JOINTS[index], errors[index], TRACKING_TOLERANCE_RAD))


class TrackingMean:
    """Mean absolute controller error; raw hardware bound remains independent."""
    def __init__(self):
        self.samples = deque(maxlen=5)

    def observe(self, row):
        stamp = row['controller_stamp_unix_s']
        if self.samples and stamp <= self.samples[-1][0]:
            return None
        while self.samples and stamp - self.samples[0][0] > .15:
            self.samples.popleft()
        self.samples.append((stamp, row['error_positions_rad']))
        means = [sum(abs(errors[i]) for _, errors in self.samples) / len(self.samples)
                 for i in range(len(JOINTS))]
        index = max(range(len(JOINTS)), key=means.__getitem__)
        if means[index] > TRACKING_TOLERANCE_RAD:
            return ('Controller mean tracking error exceeded: %s %.6f rad, limit %.6f rad '
                    '(%d samples, maximum 150 ms window)' %
                    (JOINTS[index], means[index], TRACKING_TOLERANCE_RAD, len(self.samples)))
        return None
