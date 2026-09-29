"""Build bounded rest-to-rest joint-space legs from a planned route.

Straight connections between selected route points change the interpolation;
every resulting leg must be independently collision checked before execution.
"""
from copy import deepcopy
from dataclasses import dataclass
import math

from .core import AgentError, SEGMENT_EXCURSION_RAD, distance, vector

SEGMENT_TIMEOUT = 15.0
ARRIVAL_TOLERANCE = .005
STOPPED_SPEED = .003
STOP_DWELL = .5
SETUP_DRIFT = .0005


@dataclass
class SegmentedRoute:
    segments: list
    description: str


def make_segments(trajectory):
    points = trajectory.joint_trajectory.points
    if len(points) < 2:
        raise AgentError('Segmented route requires at least two points')
    if getattr(getattr(trajectory, 'multi_dof_joint_trajectory', None), 'points', []):
        raise AgentError('Segmented execution does not support multi-DOF trajectories')
    # Reserve 0.01 rad for measured endpoint error and start alignment.
    step = SEGMENT_EXCURSION_RAD - .01
    anchors = [vector(points[0].positions)]
    previous = anchors[0]
    for point in points[1:]:
        q = vector(point.positions)
        if distance(q, anchors[-1]) > step:
            if distance(previous, anchors[-1]) > 1e-9:
                anchors.append(previous)
            origin = anchors[-1]
            count = math.ceil(distance(q, origin) / step)
            for index in range(1, count):
                anchors.append(tuple(a+(b-a)*index/count for a, b in zip(origin, q)))
            if distance(q, anchors[-1]) > step+1e-9:
                raise AgentError('Could not bound segment displacement')
        previous = q
        if len(anchors) > 128:
            raise AgentError('Route exceeds 128-segment budget')
    if distance(previous, anchors[-1]) > 1e-9:
        anchors.append(previous)
    if len(anchors) == 1:
        anchors.append(previous)
    segments = []
    for start, goal in zip(anchors, anchors[1:]):
        segment = deepcopy(trajectory)
        segments.append(retime_leg(segment, start, goal))
    if len(segments) > 128:
        raise AgentError('Route exceeds 128-segment budget')
    return segments


def retime_leg(trajectory, start, goal):
    delta = distance(start, goal)
    if delta > SEGMENT_EXCURSION_RAD:
        raise AgentError('Aligned segment exceeds per-step excursion allowance')
    # Quintic smoothstep: peak speed 1.875*d/T; acceleration (10/sqrt(3))*d/T².
    duration = max(.5, 1.875*delta/.02, math.sqrt((10/math.sqrt(3))*delta/.03))
    if duration > SEGMENT_TIMEOUT:
        raise AgentError('Segment exceeds duration budget')
    points = trajectory.joint_trajectory.points
    first, last = deepcopy(points[0]), deepcopy(points[-1])
    for p, q, seconds in ((first, start, 0.), (last, goal, duration)):
        p.positions = list(vector(q))
        p.velocities, p.accelerations = [0.]*7, [0.]*7
        if hasattr(p, 'effort'):
            p.effort = []
        nanos = math.ceil(seconds*1e9)
        p.time_from_start.sec, p.time_from_start.nanosec = divmod(nanos, 10**9)
    trajectory.joint_trajectory.points = [first, last]
    return trajectory


def require_stopped_at(state, expected, width):
    if distance(state['joints_rad'], expected) > ARRIVAL_TOLERANCE:
        raise AgentError('Segment state differs from checked route; stop and re-plan')
    if abs(state['gripper_width_m']-width) > .001:
        raise AgentError('Gripper changed between segments; stop and re-plan')
    if max(map(abs, vector(state['velocities_rad_s']))) > STOPPED_SPEED:
        raise AgentError('Arm is not stationary at segment boundary; stop and re-plan')


def fixed_frame_transforms(description):
    """Return base_link-from-frame transforms for URDF fixed-connected frames.

    Never infer world == base_link or traverse an articulated joint.
    """
    import xml.etree.ElementTree as ET
    import numpy as np
    transforms = {'base_link': np.eye(4)}
    if not description:
        return transforms
    try:
        root = ET.fromstring(description)
        edges = {}
        for joint in root.findall('joint'):
            if joint.get('type') != 'fixed':
                continue
            parent, child = joint.find('parent').get('link'), joint.find('child').get('link')
            origin = joint.find('origin')
            xyz = [float(v) for v in (origin.get('xyz', '0 0 0') if origin is not None else '0 0 0').split()]
            rpy = [float(v) for v in (origin.get('rpy', '0 0 0') if origin is not None else '0 0 0').split()]
            if len(xyz) != 3 or len(rpy) != 3 or not np.isfinite(xyz+rpy).all():
                raise ValueError('Invalid fixed joint origin')
            r, p, y = rpy
            cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
            t = np.eye(4)
            t[:3, :3] = [[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                          [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                          [-sp, cp*sr, cp*cr]]
            t[:3, 3] = xyz
            edges.setdefault(parent, []).append((child, t))
            edges.setdefault(child, []).append((parent, np.linalg.inv(t)))
        pending = ['base_link']
        while pending:
            frame = pending.pop()
            for child, t in edges.get(frame, []):
                if child not in transforms:
                    transforms[child] = transforms[frame] @ t
                    pending.append(child)
    except (ET.ParseError, AttributeError, TypeError, ValueError) as error:
        raise AgentError('Cannot resolve obstacle fixed frames: ' + str(error)) from error
    return transforms


def pose_transform(pose):
    import numpy as np
    if pose is None:  # Older CollisionObject messages have no object-level pose.
        return np.eye(4)
    p, q = pose.position, pose.orientation
    xyz = [p.x, p.y, p.z]
    quat = np.array([q.x, q.y, q.z, q.w], dtype=float)
    norm = np.linalg.norm(quat)
    if not np.isfinite(xyz).all() or not np.isfinite(quat).all() or abs(norm-1) > 1e-6:
        raise AgentError('Invalid obstacle pose: expected finite position and unit quaternion')
    x, y, z, w = quat/norm
    t = np.eye(4)
    t[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                  [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                  [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    t[:3, 3] = xyz
    return t


def require_supported_scene(scene, boxes, description=None):
    """Do not shortcut a route around obstacles absent from the MuJoCo model."""
    import numpy as np
    if scene.robot_state.attached_collision_objects or scene.world.octomap.octomap.data:
        raise AgentError('Segmented preflight does not support attached objects or octomaps')
    expected = {'nero_agent_' + b['id']: b for b in boxes}
    objects = scene.world.collision_objects
    if len(objects) != len(expected) or {o.id for o in objects} != set(expected):
        raise AgentError('MoveIt obstacles differ from the segmented preflight configuration')
    transforms = fixed_frame_transforms(description)
    for obj in objects:
        box = expected[obj.id]
        frame = obj.header.frame_id.lstrip('/')
        if frame not in transforms:
            raise AgentError('Obstacle %s uses unresolved frame %r; no fixed URDF transform to base_link' %
                             (obj.id, obj.header.frame_id))
        if (obj.meshes or obj.planes
                or len(obj.primitives) != 1 or len(obj.primitive_poses) != 1
                or obj.primitives[0].type != 1):  # SolidPrimitive.BOX
            raise AgentError('Unsupported obstacle geometry for %s in %r: primitive types %s, '
                             '%d primitive poses, %d meshes, %d planes' %
                             (obj.id, obj.header.frame_id, [p.type for p in obj.primitives],
                              len(obj.primitive_poses), len(obj.meshes), len(obj.planes)))
        # base_T_box = base_T_header * header_T_object * object_T_primitive.
        t = (transforms[frame] @ pose_transform(getattr(obj, 'pose', None))
             @ pose_transform(obj.primitive_poses[0]))
        dimensions = np.array(obj.primitives[0].dimensions, dtype=float)
        if (dimensions.shape != (3,) or not np.isfinite(dimensions).all()
                or not np.allclose(dimensions, box['size_m'], atol=1e-6, rtol=0)
                or not np.allclose(t[:3, 3], box['center_m'], atol=1e-6, rtol=0)
                or not np.allclose(t[:3, :3], np.eye(3), atol=1e-6, rtol=0)):
            raise AgentError('MoveIt obstacle %s differs from preflight configuration after '
                             'transforming %r to base_link; center=%s, dimensions=%s' %
                             (obj.id, obj.header.frame_id, t[:3, 3].tolist(), dimensions.tolist()))
