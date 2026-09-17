"""ROS-independent tool contracts and the observe/act/result loop."""

from dataclasses import dataclass
from array import array
import json
import math
import re
import time

JOINTS = tuple('joint%d' % i for i in range(1, 8))
# Shared by planning and the hardware stream guard; radians per joint.
MAX_EXCURSION_RAD = 0.30
LARGE_EXCURSION_RAD = 1.20
SEGMENT_EXCURSION_RAD = 0.10


class AgentError(RuntimeError):
    pass


def strict_json(text):
    def pairs(items):
        result = {}
        for k, v in items:
            if k in result:
                raise AgentError('Duplicate JSON key: ' + k)
            result[k] = v
        return result

    def invalid(value):
        raise AgentError('Nonfinite JSON value: ' + value)
    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def vector(value, label='joint positions'):
    # ROS float64[] fields deserialize to array('d'), including MoveIt points.
    if (not isinstance(value, (list, tuple, array)) or len(value) != 7
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) for x in value)):
        raise AgentError(label + ' must contain seven finite numbers')
    return tuple(float(x) for x in value)


def distance(a, b):
    return max(abs(x - y) for x, y in zip(vector(a), vector(b)))


def excursion_detail(positions, start, limit):
    q, initial = vector(positions), vector(start)
    index = max(range(7), key=lambda i: abs(q[i] - initial[i]))
    delta = abs(q[index] - initial[index])
    return ('%s displacement %.6f rad (%.2f deg), limit %.6f rad, excess %.6f rad; '
            'start %.6f rad, position %.6f rad' %
            (JOINTS[index], delta, math.degrees(delta), limit, max(0., delta - limit),
             initial[index], q[index]))


@dataclass(frozen=True)
class Settings:
    mode: str
    named_poses: dict
    collision_boxes: tuple
    max_excursion: float = MAX_EXCURSION_RAD
    max_velocity: float = 0.08
    max_acceleration: float = 0.15
    tolerance: float = 0.005
    timeout: float = 30.0
    namespace: str = '/nero'
    mujoco_preflight: bool = False
    segmented_execution: bool = False

    @classmethod
    def parse(cls, data, require_review=True):
        allowed = {'mode', 'named_poses', 'collision_boxes', 'reviewed_hardware', 'namespace', 'motion_profile'}
        if not isinstance(data, dict) or set(data) - allowed:
            raise AgentError('Unknown configuration fields')
        if data.get('mode') not in ('mock', 'hardware'):
            raise AgentError('Configuration mode must be mock or hardware')
        profile = data.get('motion_profile', 'bounded')
        if profile not in ('bounded', 'mujoco_large', 'mujoco_segmented'):
            raise AgentError('motion_profile must be bounded, mujoco_large or mujoco_segmented')
        if require_review and data['mode'] == 'hardware' and data.get('reviewed_hardware') is not True:
            raise AgentError('Hardware requires reviewed_hardware=true after reviewing frames, tool and collision boxes')
        ns = data.get('namespace', '/nero')
        if not isinstance(ns, str) or not re.fullmatch(r'/[A-Za-z][A-Za-z0-9_]*', ns):
            raise AgentError('namespace must be a single absolute ROS namespace')
        poses = data.get('named_poses')
        if not isinstance(poses, dict) or not 1 <= len(poses) <= 20 or 'start' in poses:
            raise AgentError('Supply 1–20 named poses; start is reserved for the captured starting state')
        for name, pose in poses.items():
            if not re.fullmatch(r'[a-z][a-z0-9_]{0,39}', name) or not isinstance(pose, dict):
                raise AgentError('Invalid named pose')
            if set(pose) not in ({'joints_rad'}, {'delta_from_start_rad'},
                                  {'position_m', 'orientation_xyzw'},
                                  {'frame', 'position_m', 'orientation_xyzw'},
                                  {'position_m', 'orientation_xyzw', 'orientation_mode'},
                                  {'frame', 'position_m', 'orientation_xyzw', 'orientation_mode'}):
                raise AgentError('Each named pose needs joints_rad, delta_from_start_rad, or Cartesian position_m/orientation_xyzw')
            if 'position_m' in pose:
                if pose.get('frame', 'base_link') != 'base_link':
                    raise AgentError('Cartesian named poses must use the base_link frame')
                for field, length in (('position_m', 3), ('orientation_xyzw', 4)):
                    values = pose[field]
                    if (not isinstance(values, list) or len(values) != length or
                            any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                                for x in values)):
                        raise AgentError('%s %s must contain %d finite numbers' % (name, field, length))
                norm = math.sqrt(sum(float(x) ** 2 for x in pose['orientation_xyzw']))
                if abs(norm - 1.0) > 1e-3:
                    raise AgentError(name + ' orientation_xyzw must be a unit quaternion')
                mode = pose.get('orientation_mode', 'fixed')
                if mode not in ('fixed', 'camera_down_free_yaw'):
                    raise AgentError(name + ' has unsupported orientation_mode')
                if mode == 'camera_down_free_yaw':
                    # Tool +Z must point along base -Z; do not silently reinterpret
                    # a tilted reference quaternion as a camera-down target.
                    x, y, z, w = (float(v)/norm for v in pose['orientation_xyzw'])
                    tool_z_in_base_z = 1 - 2*(x*x + y*y)
                    if abs(tool_z_in_base_z + 1) > 1e-6:
                        raise AgentError(name + ' camera_down_free_yaw requires tool +Z pointing down')
            else:
                vector(next(iter(pose.values())), name)
        boxes = data.get('collision_boxes')
        if not isinstance(boxes, list) or not boxes:
            raise AgentError('Configure at least the table as a collision box in base_link coordinates')
        ids = set()
        for b in boxes:
            if not isinstance(b, dict) or set(b) != {'id', 'size_m', 'center_m'}:
                raise AgentError('Collision boxes require id, size_m and center_m')
            if not isinstance(b['id'], str) or not b['id'] or b['id'] in ids:
                raise AgentError('Collision object IDs must be unique nonempty strings')
            ids.add(b['id'])
            for key in ('size_m', 'center_m'):
                v = b[key]
                if (not isinstance(v, list) or len(v) != 3
                        or any(type(x) not in (float, int) or not math.isfinite(x) for x in v)):
                    raise AgentError('Collision box coordinates must be three finite numbers')
            if min(b['size_m']) <= 0:
                raise AgentError('Collision box sizes must be positive')
        return cls(data['mode'], poses, tuple(boxes), namespace=ns,
                   max_excursion=3.14 if profile == 'mujoco_segmented' else
                   LARGE_EXCURSION_RAD if profile == 'mujoco_large' else MAX_EXCURSION_RAD,
                   timeout=600.0 if profile == 'mujoco_segmented' else 90.0 if profile == 'mujoco_large' else 30.0,
                   mujoco_preflight=profile != 'bounded',
                   segmented_execution=profile == 'mujoco_segmented')

    def resolve(self, start):
        start = vector(start)
        targets = {'start': start}
        for name, spec in self.named_poses.items():
            if 'position_m' in spec:
                targets[name] = dict(spec)
                continue
            q = (vector(spec['joints_rad']) if 'joints_rad' in spec else
                 tuple(a + b for a, b in zip(start, vector(spec['delta_from_start_rad']))))
            if distance(q, start) > self.max_excursion:
                raise AgentError('Named pose %s exceeds excursion limit: %s' %
                                 (name, excursion_detail(q, start, self.max_excursion)))
            targets[name] = q
        return targets


def validate_action(value, names):
    if not isinstance(value, dict) or set(value) != {'action', 'pose', 'reason'}:
        raise AgentError('Action must contain action, pose and reason only')
    if not isinstance(value['reason'], str) or len(value['reason']) > 1000:
        raise AgentError('Action reason must be short text')
    action, pose = value['action'], value['pose']
    if action not in ('get_state', 'move_to_named_pose', 'stop', 'finish'):
        raise AgentError('Unknown action')
    if action == 'move_to_named_pose':
        if not isinstance(pose, str) or pose not in names:
            raise AgentError('Unknown named pose')
    elif pose is not None:
        raise AgentError('Only move_to_named_pose accepts a pose')
    return value


class ScriptedChooser:
    """Test-only fixed inspection/start sequence; never interprets natural language."""
    model = 'scripted-test-double'

    def choose(self, context):
        done = [e['pose'] for e in context['history'] if e.get('status') == 'executed']
        if not done:
            return dict(action='move_to_named_pose', pose='inspection', reason='Scripted outbound test')
        if len(done) == 1:
            return dict(action='move_to_named_pose', pose='start', reason='Scripted return test')
        return dict(action='finish', pose=None, reason='Scripted round trip complete')


class DirectPoseChooser:
    """Deterministic one-shot chooser; never contacts an LLM."""
    model = 'direct-named-pose'

    def __init__(self, pose):
        self.pose = pose
        self.done = False

    def choose(self, context):
        if not self.done:
            self.done = True
            return dict(action='move_to_named_pose', pose=self.pose,
                        reason='Direct named-pose request; LLM bypassed')
        return dict(action='finish', pose=None, reason='Direct named-pose move complete')


class OfflineBackend:
    """Contract test double: no ROS, collision checking, SDK or physical motion."""
    source = 'offline_test_double'
    hardware = False

    def __init__(self, settings):
        self.settings = settings
        self.q = (0.0,) * 7
        self.stopped = False

    def state(self):
        return {'source': self.source, 'joints_rad': list(self.q), 'velocities_rad_s': [0.] * 7,
                'gripper_width_m': 0.04, 'captured_at_unix': time.time()}

    def plan(self, goal):
        return {'start': list(self.q), 'goal': list(goal), 'backend': self.source,
                'collision_checked': False}

    def execute(self, plan):
        if self.stopped:
            raise AgentError('Backend has been stopped')
        self.q = vector(plan['goal'])
        return self.state()

    def stop(self):
        self.stopped = True
        return {'status': 'stopped', 'source': self.source}

    def close(self):
        pass


def run_loop(backend, chooser, settings, instruction, execute, record, confirm, max_actions=8):
    if not isinstance(instruction, str) or not 1 <= len(instruction.strip()) <= 4000:
        raise AgentError('Instruction must be 1–4000 characters')
    if type(max_actions) is not int or not 1 <= max_actions <= 32:
        raise AgentError('max-actions must be between 1 and 32')
    initial = backend.state()
    targets = settings.resolve(initial['joints_rad'])
    history = []
    record({'event': 'initial_state', 'state': initial, 'named_poses': targets})
    for number in range(max_actions):
        state = backend.state()
        action = validate_action(chooser.choose({'instruction': instruction, 'state': state,
                                                 'named_poses': targets, 'history': history}), targets)
        record({'event': 'decision', 'number': number + 1, **action})
        if action['action'] == 'finish':
            final = backend.state()
            return {'status': 'completed', 'final_state': final,
                    'returned_to_start': distance(final['joints_rad'], initial['joints_rad']) <= settings.tolerance}
        if action['action'] == 'stop':
            result = backend.stop()
            record({'event': 'stop', **result})
            return {'status': 'stopped', 'final_state': backend.state()}
        if action['action'] == 'get_state':
            result = {'status': 'observed', 'state': backend.state(), 'action': 'get_state'}
        else:
            goal = targets[action['pose']]
            plan = backend.plan(goal)
            record({'event': 'planned', 'pose': action['pose'], 'plan': plan})
            if not execute:
                return {'status': 'planned_only', 'final_state': backend.state()}
            if backend.hardware and not confirm('Plan for %s is ready. Type EXECUTE to move: ' % action['pose']):
                raise AgentError('Physical execution cancelled')
            record({'event': 'execution_pending', 'pose': action['pose']})
            achieved = backend.execute(plan)
            if distance(achieved['joints_rad'], plan['goal']) > settings.tolerance:
                raise AgentError('Measured state did not reach the requested pose')
            result = {'status': 'executed', 'pose': action['pose'], 'state': achieved}
        history.append(result)
        record({'event': 'action_result', **result})
    raise AgentError('Action budget exhausted; no further motion requested')
