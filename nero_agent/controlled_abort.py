"""Experimental vendor MOVE J hold, with bounded cancellation and observation.

No automatic damping stop, reset, disable, enable, or re-arming. This verifies
observations over a bounded interval, not a guaranteed deceleration profile.
"""
import time
from .core import AgentError, distance, vector


class ControlledAbort:
    CANCEL_TIMEOUT = 0.2
    SETTLE_TIMEOUT = 8.0
    HOLD_DWELL = 0.5
    POSITION_TOLERANCE = 0.005
    MAX_EXCURSION = 0.02
    STOPPED_SPEED = 0.02
    RECHECK_TIMEOUT = 5.0

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
        self.recheck_count = 0
        self.recheck_deadline = None
        self.longest_stable_dwell = 0.0
        self.speed_dwell_resets = 0
        self.position_dwell_resets = 0
        self.last_position_error = None
        self.last_peak_speed = None

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
            if self.phase == 'rechecking' and now >= self.recheck_deadline:
                raise AgentError(self._settling_failure('Powered hold recheck', self.RECHECK_TIMEOUT))
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
            self.last_position_error = distance(q, self.target)
            self.last_peak_speed = max(map(abs, v))
            position_ok = self.last_position_error <= self.POSITION_TOLERANCE
            speed_ok = self.last_peak_speed <= self.STOPPED_SPEED
            stationary = position_ok and speed_ok
            if self.stable_since is not None and not stationary:
                self.speed_dwell_resets += int(not speed_ok)
                self.position_dwell_resets += int(not position_ok)
            if self.phase == 'holding' and not stationary:
                self.phase = 'rechecking'
                self.recheck_count += 1
                self.recheck_deadline = now + self.RECHECK_TIMEOUT
                self.deadline = self.recheck_deadline
                self.timings['last_recheck_started_monotonic_s'] = now
                self.stable_since = None
            if stationary:
                if self.first_stable_since is None:
                    self.first_stable_since = now
                if self.stable_since is None:
                    self.stable_since = now
                self.longest_stable_dwell = max(self.longest_stable_dwell, now-self.stable_since)
                if now - self.stable_since >= self.HOLD_DWELL:
                    if self.phase == 'rechecking':
                        self.timings['last_recheck_confirmed_monotonic_s'] = now
                    self.phase = 'holding'
            else:
                self.last_nonstationary = now
                self.stable_since = None
            if self.phase != 'holding' and now >= self.deadline:
                raise AgentError(self._settling_failure('Controlled abort', self.SETTLE_TIMEOUT))
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

    def _settling_failure(self, label, timeout):
        return ('%s did not settle within %g seconds; required %g s dwell, speed <= %g rad/s, '
                'position error <= %g rad; last speed %s rad/s, last position error %s rad; '
                'longest observed dwell %.3f s, speed resets %d, position resets %d' %
                (label, timeout, self.HOLD_DWELL, self.STOPPED_SPEED, self.POSITION_TOLERANCE,
                 self.last_peak_speed, self.last_position_error, self.longest_stable_dwell,
                 self.speed_dwell_resets, self.position_dwell_resets))

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
                'settle_timeout_s': self.SETTLE_TIMEOUT,
                'stopped_speed_rad_s': self.STOPPED_SPEED,
                'position_tolerance_rad': self.POSITION_TOLERANCE,
                'longest_stable_dwell_s': self.longest_stable_dwell,
                'speed_dwell_resets': self.speed_dwell_resets,
                'position_dwell_resets': self.position_dwell_resets,
                'last_position_error_rad': self.last_position_error,
                'last_peak_speed_rad_s': self.last_peak_speed,
                'recheck_count': self.recheck_count,
                'recheck_timeout_s': self.RECHECK_TIMEOUT,
                'recheck_deadline_monotonic_s': self.recheck_deadline,
                'stable_since_monotonic_s': self.stable_since,
                'first_stable_since_monotonic_s': self.first_stable_since,
                'last_nonstationary_monotonic_s': self.last_nonstationary,
                'failure': getattr(self, 'failure', None),
                'timings': dict(self.timings),
                'physically_validated': False}
