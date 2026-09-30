"""Bounded controller-state capture; no disk writes or control decisions."""
from collections import deque
import math
import time

from .core import JOINTS, vector


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
