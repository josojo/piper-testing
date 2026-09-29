"""Diagnostics for measured arrival; does not change acceptance criteria."""
from .core import JOINTS


class SettlingDiagnostics:
    def __init__(self, goal, position_limit, speed_limit, dwell):
        self.goal = list(goal)
        self.position_limit, self.speed_limit, self.dwell = position_limit, speed_limit, dwell
        self.samples = 0
        self.position_failures = [0] * 7
        self.speed_failures = [0] * 7
        self.peak_errors = [0.] * 7
        self.peak_speeds = [0.] * 7
        self.still_since = self.settled_since = None
        self.longest_still = self.longest_settled = 0.
        self.last = None

    def observe(self, actual, now):
        errors = [abs(q-g) for q, g in zip(actual['joints_rad'], self.goal)]
        speeds = list(map(abs, actual['velocities_rad_s']))
        self.samples += 1
        for i in range(7):
            self.position_failures[i] += errors[i] > self.position_limit
            self.speed_failures[i] += speeds[i] > self.speed_limit
            self.peak_errors[i] = max(self.peak_errors[i], errors[i])
            self.peak_speeds[i] = max(self.peak_speeds[i], speeds[i])
        still = max(speeds) <= self.speed_limit
        settled = still and max(errors) <= self.position_limit
        for condition, since, longest in ((still, 'still_since', 'longest_still'),
                                          (settled, 'settled_since', 'longest_settled')):
            if condition:
                if getattr(self, since) is None:
                    setattr(self, since, now)
                setattr(self, longest, max(getattr(self, longest), now-getattr(self, since)))
            else:
                setattr(self, since, None)
        self.last = {'joints_rad': list(actual['joints_rad']),
                     'velocities_rad_s': list(actual['velocities_rad_s']),
                     'position_errors_rad': errors}

    def result(self):
        return {'samples': self.samples, 'goal_rad': self.goal,
                'position_limit_rad': self.position_limit, 'speed_limit_rad_s': self.speed_limit,
                'required_dwell_s': self.dwell,
                'longest_standstill_s': self.longest_still,
                'longest_at_goal_and_still_s': self.longest_settled,
                'last_sample': self.last,
                'joints': {name: {'position_failure_samples': self.position_failures[i],
                                  'speed_failure_samples': self.speed_failures[i],
                                  'peak_position_error_rad': self.peak_errors[i],
                                  'peak_speed_rad_s': self.peak_speeds[i]}
                           for i, name in enumerate(JOINTS)}}
