"""Shared, model-derived gripping TCP. All quaternions here use ROS xyzw order."""
import json
import math
from pathlib import Path

CONFIG = Path(__file__).with_name('tool_frame.json')


def multiply(a, b):
    x, y, z, w = a
    X, Y, Z, W = b
    return (w*X+x*W+y*Z-z*Y, w*Y-x*Z+y*W+z*X,
            w*Z+x*Y-y*X+z*W, w*W-x*X-y*Y-z*Z)


def rotate(q, v):
    return multiply(multiply(q, (*v, 0.0)), (-q[0], -q[1], -q[2], q[3]))[:3]


def from_rpy(rpy):
    r, p, y = (v/2 for v in rpy)
    return multiply(multiply((0, 0, math.sin(y), math.cos(y)),
                             (0, math.sin(p), 0, math.cos(p))),
                    (math.sin(r), 0, 0, math.cos(r)))


def to_rpy(q):
    x, y, z, w = q
    return (math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
            math.asin(max(-1., min(1., 2*(w*y-z*x)))),
            math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z)))


def compose(a, b):
    p, q = a
    t, r = b
    offset = rotate(q, t)
    return tuple(x+y for x, y in zip(p, offset)), multiply(q, r)


def definition():
    return json.loads(CONFIG.read_text())


def tcp_in_link7(root):
    """Resolve the vendor fixed chain, then add the shared contact-center frame."""
    spec = definition()
    by_child = {j.find('child').get('link'): j for j in root.findall('joint')}
    chain = []
    link = spec['reference_link']
    seen = set()
    while link != 'link7':
        if link in seen or link not in by_child:
            raise ValueError('TCP reference must have a fixed chain to link7')
        seen.add(link)
        joint = by_child[link]
        if joint.get('type') != 'fixed':
            raise ValueError('TCP reference must not depend on gripper opening')
        origin = joint.find('origin')
        xyz = tuple(map(float, origin.get('xyz', '0 0 0').split())) if origin is not None else (0.,)*3
        rpy = tuple(map(float, origin.get('rpy', '0 0 0').split())) if origin is not None else (0.,)*3
        chain.append((xyz, from_rpy(rpy)))
        link = joint.find('parent').get('link')
    transform = ((0.,)*3, (0., 0., 0., 1.))
    for part in reversed(chain):
        transform = compose(transform, part)
    return compose(transform, (spec['position_m'], from_rpy(spec['rpy_rad'])))


def xacro_mappings(root):
    position, quaternion = tcp_in_link7(root)
    return {'tcp_offset_xyz': ' '.join(format(v, '.16g') for v in position),
            'tcp_offset_rpy': ' '.join(format(v, '.16g') for v in to_rpy(quaternion))}


def migrate_flange_pose(pose, transform):
    """Preserve the old reference flange pose; free-yaw sets cannot be preserved."""
    result = dict(pose)
    position, quaternion = compose((pose['position_m'], pose['orientation_xyzw']), transform)
    result.update(position_m=list(position), orientation_xyzw=list(quaternion))
    if result.get('orientation_mode') == 'camera_down_free_yaw':
        result['orientation_mode'] = 'fixed'
    return result
