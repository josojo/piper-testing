"""Bounds checks on MoveIt output; planning and interpolation belong to MoveIt/controller."""
import math
import xml.etree.ElementTree as ET
from .core import AgentError, JOINTS, distance, vector, excursion_detail


def validate_model_bounds(description, start, goal):
    """Reject states that MoveIt could otherwise clamp to URDF bounds."""
    try:
        root = ET.fromstring(description)
        bounds = {}
        for name in JOINTS:
            joint = root.find('./joint[@name="%s"]' % name)
            if joint is None or joint.get('type') != 'revolute':
                raise ValueError('Expected bounded revolute joint ' + name)
            limit = joint.find('limit')
            low, high = float(limit.get('lower')), float(limit.get('upper'))
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError('Invalid bounds for ' + name)
            bounds[name] = (low, high)
    except (ET.ParseError, AttributeError, TypeError, ValueError) as error:
        raise AgentError('Cannot validate loaded robot model limits: ' + str(error)) from error
    for label, positions in (('Measured start', start), ('Requested goal', goal)):
        for name, position in zip(JOINTS, vector(positions)):
            low, high = bounds[name]
            if not low <= position <= high:
                raise AgentError('%s %s %.6f rad is outside loaded robot model limits '
                                 '[%.6f, %.6f] rad. Verify joint conventions and the robot model '
                                 'before planning; no clamping or automatic recovery motion.' %
                                 (label, name, position, low, high))


def validate_trajectory(trajectory, start, goal, initial, settings):
    if list(trajectory.joint_names) != list(JOINTS) or len(trajectory.points) < 2:
        raise AgentError('MoveIt must return a complete seven-joint timed trajectory')
    previous = -1.0
    peak_v, peak_a = 0.0, 0.0
    peak_excursion, peak_q, peak_time = -1.0, None, None
    timing_error = None
    for index, point in enumerate(trajectory.points):
        q = vector(point.positions)
        v, a = vector(point.velocities, 'trajectory velocity'), vector(point.accelerations, 'trajectory acceleration')
        stamp = point.time_from_start.sec + point.time_from_start.nanosec * 1e-9
        reason = None
        if not math.isfinite(stamp):
            reason = 'Trajectory timestamp is nonfinite'
        elif stamp < 0:
            reason = 'Trajectory timestamp is negative'
        elif stamp <= previous:
            reason = 'Trajectory timestamps do not strictly increase'
        elif stamp > settings.timeout:
            reason = 'Trajectory exceeds execution timeout'
        if reason and timing_error is None:
            timing_error = ('%s: point index %d (zero-based), timestamp %.6f s, '
                            'previous timestamp %.6f s, timeout %.6f s' %
                            (reason, index, stamp, previous, settings.timeout))
        excursion = distance(q, initial)
        if excursion > peak_excursion:
            peak_excursion, peak_q, peak_time = excursion, q, stamp
        peak_v = max(peak_v, max(map(abs, v)))
        peak_a = max(peak_a, max(map(abs, a)))
        previous = stamp
    if timing_error is not None:
        raise AgentError('%s; final timestamp (planned duration if timing is valid) %.6f s; '
                         'maximum joint excursion: %s; peak at trajectory time %.6f s' %
                         (timing_error, previous,
                          excursion_detail(peak_q, initial, settings.max_excursion), peak_time))
    if peak_excursion > settings.max_excursion + 1e-6:
        raise AgentError('Trajectory leaves the captured-start excursion envelope: %s; '
                         'peak at trajectory time %.6f s' %
                         (excursion_detail(peak_q, initial, settings.max_excursion), peak_time))
    if peak_v > settings.max_velocity + 1e-6 or peak_a > settings.max_acceleration + 1e-6:
        raise AgentError('Trajectory exceeds experiment velocity/acceleration limits')
    if distance(trajectory.points[0].positions, start) > settings.tolerance:
        raise AgentError('Trajectory starts at the wrong state')
    if distance(trajectory.points[-1].positions, goal) > settings.tolerance:
        endpoint = vector(trajectory.points[-1].positions)
        index = max(range(7), key=lambda i: abs(endpoint[i] - goal[i]))
        raise AgentError('Trajectory ends at the wrong goal: %s planned %.6f rad, requested %.6f rad '
                         '(error %.6f rad, tolerance %.6f rad)' %
                         (JOINTS[index], endpoint[index], goal[index],
                          abs(endpoint[index] - goal[index]), settings.tolerance))
    return {'points': len(trajectory.points), 'duration_s': previous,
            'peak_excursion_rad': peak_excursion,
            'max_excursion_rad': settings.max_excursion,
            'peak_velocity_rad_s': peak_v, 'peak_acceleration_rad_s2': peak_a}
