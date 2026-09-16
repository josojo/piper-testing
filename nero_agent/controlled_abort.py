"""Experimental vendor MOVE J hold, with bounded cancellation and observation.

No automatic damping stop, reset, disable, enable, or re-arming. This verifies
observations over a bounded interval, not a guaranteed deceleration profile.
"""
import time
from .core import AgentError, distance, vector


class ControlledAbort:
    CANCEL_TIMEOUT = 0.2
    SETTLE_TIMEOUT = 5.0
    HOLD_DWELL = 1.0
    POSITION_TOLERANCE = 0.005
    MAX_EXCURSION = 0.02
    STOPPED_SPEED = 0.01

    def __init__(self, hardware, block_stream, cancel_controller, clock=time.monotonic):
        self.hardware, self.block_stream = hardware, block_stream
        self.cancel_controller, self.clock = cancel_controller, clock
        self.timings = {}
        self.phase = 'idle'
        self.target = None
        self.stable_since = None
        self.first_stable_since = None
        self.last_nonstationary = None
        self.cancellation = 'not_requested'
        self.reason = None
        self.samples = 0
        self.last_state = None
        self.last_source_stamp = None

    def start(self, reason):
        if self.phase != 'idle':
            return self.result()  # Idempotent: never chase the latest position.
        self.timings['abort_requested_monotonic_s'] = self.clock()
        self.block_stream()
        self.reason = reason
        self.phase = 'cancelling'
        self.deadline = self.clock() + self.CANCEL_TIMEOUT
        self.future = None
        try:
            self.future = self.cancel_controller()
            self.cancellation = 'pending'
        except Exception as error:
            self.cancellation = 'failed: ' + str(error)
        return self.result()

    def fail(self, error):
        self.phase = 'failed'
        self.failure = str(error)
        # Do not change mode or release torque as an implicit fallback.

    def tick(self):
        if self.phase in ('idle', 'failed'):
            return
        try:
            now = self.clock()
            if self.phase == 'cancelling':
                if self.future is not None and not self.future.done() and now < self.deadline:
                    return
                if self.future is not None:
                    if self.future.done():
                        response = self.future.result()
                        self.cancellation = ('acknowledged' if response.return_code == 0 else
                                             'not_confirmed: code %s' % response.return_code)
                    else:
                        self.cancellation = 'timeout'
                self.timings['cancellation_resolved_monotonic_s'] = self.clock()
                self._hold(now)
                return
            state = self.hardware.read(require_enabled=True)
            q, v = vector(state.joints_rad), vector(state.velocities_rad_s)
            # The checked reader verifies freshness. Count only advancing source
            # snapshots, not repeated reads of the same SDK cache.
            stamp = min(state.feedback_timestamps)
            if self.last_source_stamp is not None and stamp <= self.last_source_stamp:
                if self.phase != 'holding' and now >= self.deadline:
                    raise AgentError('No advancing feedback during hold verification')
                return
            self.last_source_stamp = stamp
            self.last_state = {'joints_rad': list(q), 'velocities_rad_s': list(v),
                               'enabled': list(state.enabled), 'source_stamp': stamp}
            self.samples += 1
            if distance(q, self.target) > self.MAX_EXCURSION:
                raise AgentError('Controlled abort exceeded %.3f rad hold excursion' % self.MAX_EXCURSION)
            stationary = (distance(q, self.target) <= self.POSITION_TOLERANCE and
                          max(map(abs, v)) <= self.STOPPED_SPEED)
            if self.phase == 'holding' and not stationary:
                raise AgentError('Powered hold lost after standstill verification')
            if stationary:
                if self.first_stable_since is None:
                    self.first_stable_since = now
                if self.stable_since is None:
                    self.stable_since = now
                if now - self.stable_since >= self.HOLD_DWELL:
                    self.phase = 'holding'
            else:
                self.last_nonstationary = now
                self.stable_since = None
            if self.phase != 'holding' and now >= self.deadline:
                raise AgentError('Controlled abort did not settle within 5 seconds')
        except Exception as error:
            # Even cancellation transport failure must not skip the fresh-state
            # holding attempt. The gate is already latched closed.
            if self.phase == 'cancelling':
                self.cancellation = 'failed: ' + str(error)
                try:
                    self._hold(self.clock())
                except Exception as hold_error:
                    self.fail(hold_error)
            else:
                self.fail(error)

    def _hold(self, now):
        self.phase = 'sending_hold'
        self.timings['hold_feedback_read_started_monotonic_s'] = self.clock()
        state = self.hardware.read(require_enabled=True)
        self.timings['hold_feedback_read_finished_monotonic_s'] = self.clock()
        self.target = vector(state.joints_rad)
        self.timings['move_j_started_monotonic_s'] = self.clock()
        self.hardware.robot.move_j(list(self.target))
        self.timings['move_j_returned_monotonic_s'] = self.clock()
        self.last_source_stamp = min(state.feedback_timestamps)
        self.phase = 'settling'
        self.deadline = now + self.SETTLE_TIMEOUT

    def result(self):
        return {'status': self.phase, 'reason': self.reason,
                'controller_cancellation': self.cancellation,
                'hold_target_rad': self.target, 'observed_state': self.last_state,
                'feedback_samples': self.samples,
                'hold_dwell_s': self.HOLD_DWELL,
                'stable_since_monotonic_s': self.stable_since,
                'first_stable_since_monotonic_s': self.first_stable_since,
                'last_nonstationary_monotonic_s': self.last_nonstationary,
                'failure': getattr(self, 'failure', None),
                'timings': dict(self.timings),
                'physically_validated': False}
