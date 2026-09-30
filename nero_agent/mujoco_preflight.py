"""Conservative geometric preflight, not a calibrated dynamics simulation.

Use the loaded MoveIt URDF, measured gripper and configured boxes. Enclose
collision meshes in boxes and bound the controller's quintic interpolation
with Bezier control points. Missing geometry or inconclusive clearance fails.
"""
import hashlib
import itertools
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .core import (AgentError, JOINTS, FLOOR_SLOW_BAND_M, FLOOR_MIN_Z_M,
                   GRIPPER_PREFLIGHT_ALLOWANCE_M, TRACKING_TOLERANCE_RAD)
from nero_planner.model import STRUCTURAL_BODY_PAIRS, box_clearances


def bezier_segment(first, last, dt):
    q0, q1 = np.array(first.positions), np.array(last.positions)
    v0, v1 = np.array(first.velocities), np.array(last.velocities)
    a0, a1 = np.array(first.accelerations), np.array(last.accelerations)
    b = np.empty((6, 7))
    b[0], b[5] = q0, q1
    b[1], b[4] = q0 + dt*v0/5, q1 - dt*v1/5
    b[2] = dt*dt*a0/20 + 2*b[1] - b[0]
    b[3] = dt*dt*a1/20 + 2*b[4] - b[5]
    return b


def split(b):
    rows = [b]
    while len(rows[-1]) > 1:
        rows.append((rows[-1][:-1] + rows[-1][1:])/2)
    return np.array([r[0] for r in rows]), np.array([r[-1] for r in rows[::-1]])


def retime_interpolation(trajectory, settings, allow_speedup=False):
    """Uniformly retime the full quintic curve, preserving its geometric path."""
    peak_v = peak_a = 0.
    for first, last in zip(trajectory.points, trajectory.points[1:]):
        dt = ((last.time_from_start.sec-first.time_from_start.sec)
              + (last.time_from_start.nanosec-first.time_from_start.nanosec)*1e-9)
        if not math.isfinite(dt) or dt <= 0:
            raise AgentError('Interpolation retiming requires increasing timestamps')
        b = bezier_segment(first, last, dt)
        velocity = 5*np.diff(b, axis=0)/dt
        acceleration = 4*np.diff(velocity, axis=0)/dt
        if not np.isfinite(velocity).all() or not np.isfinite(acceleration).all():
            raise AgentError('Nonfinite interpolation derivatives')
        # Tighten the convex-hull bounds without losing their conservatism.
        pieces = [(velocity, acceleration)]
        for _ in range(4):
            pieces = [pair for v, a in pieces for pair in zip(split(v), split(a))]
        peak_v = max(peak_v, *(float(np.max(np.abs(v))) for v, _ in pieces))
        peak_a = max(peak_a, *(float(np.max(np.abs(a))) for _, a in pieces))
    end = trajectory.points[-1].time_from_start
    duration = end.sec + end.nanosec*1e-9
    scale = max(.1 / duration if allow_speedup else 1., peak_v/settings.max_velocity,
                math.sqrt(peak_a/settings.max_acceleration))
    if scale > 1. or (allow_speedup and scale < 1.):
        scale *= 1.02  # Margin for timestamp rounding and bound evaluation.
        end = trajectory.points[-1].time_from_start
        if (end.sec + end.nanosec*1e-9)*scale > settings.timeout:
            raise AgentError('Interpolation retiming exceeds execution timeout')
        for point in trajectory.points:
            stamp = point.time_from_start
            nanos = round((stamp.sec*10**9 + stamp.nanosec)*scale)
            stamp.sec, stamp.nanosec = divmod(nanos, 10**9)
            point.velocities = [v/scale for v in point.velocities]
            point.accelerations = [a/scale**2 for a in point.accelerations]
    return {'interpolation_time_scale': scale,
            'original_interpolation_velocity_bound_rad_s': peak_v,
            'original_interpolation_acceleration_bound_rad_s2': peak_a}


class CollisionScene:
    def __init__(self, description, boxes, gripper):
        if not math.isfinite(gripper) or not 0 <= gripper <= .1:
            raise AgentError('MuJoCo requires a measured gripper opening in [0, 0.1] m')
        root = ET.fromstring(description)
        self.inertial_placeholders = []
        # The vendor's virtual opening joint has a geometry-free child with no
        # inertia. MoveIt accepts it, but MuJoCo requires positive inertia even
        # for kinematic mj_forward checks. Match prepare_nero_mujoco's existing
        # compatibility adjustment, only in this private geometric-check model.
        virtual = root.find('./link[@name="gripper_link"]')
        opening = root.find('./joint[@name="gripper"]')
        if virtual is not None and virtual.find('inertial') is None:
            if (virtual.find('collision') is not None or virtual.find('visual') is not None
                    or opening is None or opening.get('type') != 'prismatic'
                    or opening.find('child') is None
                    or opening.find('child').get('link') != 'gripper_link'):
                raise AgentError('Unrecognized massless gripper link; cannot apply kinematic placeholder')
            inertial = ET.SubElement(virtual, 'inertial')
            ET.SubElement(inertial, 'mass', {'value': '0.001'})
            ET.SubElement(inertial, 'inertia', {
                'ixx': '1e-8', 'iyy': '1e-8', 'izz': '1e-8',
                'ixy': '0', 'ixz': '0', 'iyz': '0'})
            self.inertial_placeholders.append('gripper_link')
        # Unique collision names make silent loss of URDF geometry detectable.
        expected = []
        required_links = {'base_link', *(f'link{i}' for i in range(1, 8)),
                          'gripper_link1', 'gripper_link2'}
        covered = {link.get('name') for link in root.findall('link') if link.findall('collision')}
        if not required_links <= covered:
            raise AgentError('MuJoCo preflight requires collision geometry on every arm link and finger')
        for link in root.findall('link'):
            for visual in list(link.findall('visual')):
                link.remove(visual)
            for index, collision in enumerate(link.findall('collision')):
                name = 'preflight_%s_%d' % (link.get('name'), index)
                collision.set('name', name)
                expected.append(name)
        for mesh in root.findall('.//mesh'):
            path = mesh.get('filename')
            if path.startswith('package://'):
                from ament_index_python.packages import get_package_share_directory
                package, relative = path[len('package://'):].split('/', 1)
                path = str(Path(get_package_share_directory(package)) / relative)
            elif path.startswith('file://'):
                path = path[len('file://'):]
            if not Path(path).is_absolute() or not Path(path).is_file():
                raise AgentError('MuJoCo collision mesh must resolve to an existing absolute path: ' + path)
            mesh.set('filename', path)
        extension = root.find('mujoco')
        if extension is None:
            extension = ET.SubElement(root, 'mujoco')
        compiler = extension.find('compiler')
        if compiler is None:
            compiler = ET.SubElement(extension, 'compiler')
        compiler.set('discardvisual', 'true')
        compiler.set('fusestatic', 'true')
        compiler.set('strippath', 'false')
        self.model = m = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
        self.data = mujoco.MjData(m)
        if not expected or set(expected) != {m.geom(i).name for i in range(m.ngeom)}:
            raise AgentError('MuJoCo did not retain exactly the loaded URDF collision geometry')
        ids = [m.joint(n).id for n in JOINTS]
        if any(m.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE or not m.jnt_limited[i] for i in ids):
            raise AgentError('MuJoCo requires seven bounded arm hinges')
        self.qadr = m.jnt_qposadr[ids]
        self.ranges = m.jnt_range[ids]
        allowed = set(JOINTS) | {'gripper', 'gripper_joint1', 'gripper_joint2'}
        if {m.joint(i).name for i in range(m.njnt)} != allowed:
            raise AgentError('Unexpected MuJoCo movable joint set')
        for name, q in (('gripper', gripper), ('gripper_joint1', gripper/2),
                        ('gripper_joint2', -gripper/2)):
            self.data.qpos[m.jnt_qposadr[m.joint(name).id]] = q
        self.offsets, sizes = [], []
        for i in range(m.ngeom):
            if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = m.geom_dataid[i]
                vertices = m.mesh_vert[m.mesh_vertadr[mesh]:m.mesh_vertadr[mesh]+m.mesh_vertnum[mesh]]
                low, high = vertices.min(axis=0), vertices.max(axis=0)
                self.offsets.append((low+high)/2)
                sizes.append((high-low)/2 + .003)
            elif m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
                self.offsets.append(np.zeros(3))
                sizes.append(m.geom_size[i] + .003)
            else:
                raise AgentError('Unsupported collision primitive in MuJoCo preflight')
        self.offsets = np.array(self.offsets)
        self.names = [m.geom(i).name for i in range(m.ngeom)] + [b['id'] for b in boxes]
        bodies = [m.body(m.geom_bodyid[i]).name for i in range(m.ngeom)]
        from .segmented import fixed_frame_transforms
        parents = {joint.find('child').get('link') for joint in root.findall('joint')}
        roots = {link.get('name') for link in root.findall('link')} - parents
        frames = fixed_frame_transforms(description)
        if len(roots) != 1 or next(iter(roots)) not in frames:
            raise AgentError('MuJoCo requires a known fixed transform from model root to base_link')
        root_from_base = np.linalg.inv(frames[next(iter(roots))])
        self.base_from_root = frames[next(iter(roots))]
        self.obstacles = np.array([b['center_m'] for b in boxes], dtype=float).reshape(-1, 3)
        self.obstacles = self.obstacles @ root_from_base[:3, :3].T + root_from_base[:3, 3]
        self.obstacle_rotations = np.tile(root_from_base[:3, :3], (len(boxes), 1, 1))
        self.sizes = np.array(sizes + [np.array(b['size_m'])/2 for b in boxes])
        pairs = []
        for a, b in itertools.combinations(range(len(self.names)), 2):
            if a >= m.ngeom:
                continue
            if b < m.ngeom:
                if bodies[a] == bodies[b] or frozenset((bodies[a], bodies[b])) in STRUCTURAL_BODY_PAIRS:
                    continue
            # Fixed base/table mounting interface only, not the moving arm.
            elif bodies[a] == 'world' and self.names[b] == 'table':
                continue
            pairs.append((a, b))
        if not pairs:
            raise AgentError('MuJoCo preflight has no collision pairs')
        self.first, self.second = np.array(pairs).T
        ancestry = np.zeros((len(self.names), 7), dtype=bool)
        joint_bodies = list(m.jnt_bodyid[ids])
        opening_bodies = {m.jnt_bodyid[m.joint(name).id]: factor for name, factor in
                          (('gripper', 1.), ('gripper_joint1', .5), ('gripper_joint2', .5))}
        for g in range(m.ngeom):
            body = m.geom_bodyid[g]
            while body:
                # Inflate descendants of opening joints to cover setup drift and
                # live gripper feedback tolerance, in every spatial direction.
                self.sizes[g] += opening_bodies.get(body, 0.) * GRIPPER_PREFLIGHT_ALLOWANCE_M
                if body in joint_bodies:
                    ancestry[g, joint_bodies.index(body)] = True
                body = m.body_parentid[body]
        radii = np.zeros((len(self.names), 7))
        for g in range(m.ngeom):
            body = m.geom_bodyid[g]
            radius = (np.linalg.norm(m.geom_pos[g]) + np.linalg.norm(self.offsets[g])
                      + np.linalg.norm(self.sizes[g]))
            while body:
                if body in joint_bodies:
                    column = joint_bodies.index(body)
                    radii[g, column] = radius + np.linalg.norm(m.jnt_pos[ids[column]])
                for jid in range(m.body_jntadr[body], m.body_jntadr[body]+m.body_jntnum[body]):
                    if m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_SLIDE:
                        radius += max(abs(m.jnt_range[jid]))
                    elif m.jnt_type[jid] == mujoco.mjtJoint.mjJNT_HINGE:
                        radius += 2*np.linalg.norm(m.jnt_pos[jid])
                radius += np.linalg.norm(m.body_pos[body])
                body = m.body_parentid[body]
        # Floor checks include every collision geom except the fixed mounting base.
        # Arm links and all fixed/movable gripper geometry remain included.
        self.floor_geoms = np.array([i for i in range(m.ngeom)
                                    if not self.names[i].startswith('preflight_base_link_')])
        self.floor_motion = radii[self.floor_geoms]
        # Common ancestor rotations preserve relative separation.
        self.motion = np.where(ancestry[self.first], radii[self.first], radii[self.second])
        self.motion *= ancestry[self.first] ^ ancestry[self.second]

    def distances(self, q):
        self.data.qpos[self.qadr] = q
        mujoco.mj_forward(self.model, self.data)
        rotations = self.data.geom_xmat.reshape(-1, 3, 3)
        centers = self.data.geom_xpos + np.einsum('nij,nj->ni', rotations, self.offsets)
        centers = np.concatenate((centers, self.obstacles))
        rotations = np.concatenate((rotations, self.obstacle_rotations))
        return box_clearances(centers, rotations, self.sizes, self.first, self.second)


    def floor_heights(self):
        """Lowest padded box points above base_link z=0 after distances(q)."""
        rotations = self.data.geom_xmat.reshape(-1, 3, 3)
        centers = self.data.geom_xpos + np.einsum('nij,nj->ni', rotations, self.offsets)
        transform = self.base_from_root
        centers = centers @ transform[:3, :3].T + transform[:3, 3]
        rotations = np.einsum('ij,njk->nik', transform[:3, :3], rotations)
        heights = centers[:, 2] - np.einsum('ni,ni->n', abs(rotations[:, 2, :]),
                                           self.sizes[:self.model.ngeom])
        return heights[self.floor_geoms]


def validate(description, trajectory, before, initial, settings):
    """Reject on collision, bounds, or exhausted continuous-check budget."""
    try:
        if list(trajectory.joint_names) != list(JOINTS) or len(trajectory.points) < 2:
            raise AgentError('MuJoCo requires a complete seven-joint trajectory')
        scene = CollisionScene(description, settings.collision_boxes, before['gripper_width_m'])
        count, minimum = 0, float('inf')
        minimum_floor = float('inf')
        # Cover a possible controller initial transition from fresh feedback.
        # Include tracking tolerance in each geometric envelope as well.
        padding = TRACKING_TOLERANCE_RAD
        low_limit = np.maximum(scene.ranges[:, 0], np.array(initial)-settings.max_excursion)
        high_limit = np.minimum(scene.ranges[:, 1], np.array(initial)+settings.max_excursion)
        points = trajectory.points
        if points[0].time_from_start.sec != 0 or points[0].time_from_start.nanosec != 0:
            raise AgentError('MuJoCo preflight requires a trajectory starting at time zero')
        for interval, (first, last) in enumerate(zip(points, points[1:])):
            dt = ((last.time_from_start.sec-first.time_from_start.sec)
                  + (last.time_from_start.nanosec-first.time_from_start.nanosec)*1e-9)
            if not math.isfinite(dt) or dt <= 0:
                raise AgentError('MuJoCo preflight requires strictly increasing trajectory times')
            start_time = first.time_from_start.sec + first.time_from_start.nanosec*1e-9
            b = bezier_segment(first, last, dt)
            velocity = 5*np.diff(b, axis=0)/dt
            acceleration = 4*np.diff(velocity, axis=0)/dt
            pending = [(b, velocity, acceleration, dt, 0, start_time)]
            while pending:
                b, velocity, acceleration, span, depth, at = pending.pop()
                count += 1
                if count > 20000:
                    raise AgentError('MuJoCo preflight exhausted its clearance-validation budget; '
                                     'waypoint interval %d, time [%.9f, %.9f] s, depth %d; '
                                     'last checked interval: %s' %
                                     (interval, at, at+span, depth, last_diagnostic))
                if not all(np.isfinite(values).all() for values in (b, velocity, acceleration)):
                    raise AgentError('Nonfinite controller interpolation')
                left, right = split(b)
                q = left[-1]
                lower, upper = b.min(axis=0), b.max(axis=0)
                distances = scene.distances(q)
                worst = int(np.argmin(distances))
                interpolation_loss = scene.motion @ np.maximum(q-lower, upper-q)
                tracking_loss = scene.motion @ np.full(7, padding)
                bounds = distances-interpolation_loss-tracking_loss
                clearance = float(np.min(bounds))
                floor_bound = float('inf')
                floor_refine = False
                if settings.floor_guard:
                    heights = scene.floor_heights()
                    floor_loss = scene.floor_motion @ (np.maximum(q-lower, upper-q) + padding)
                    floor_bound = float(np.min(heights-floor_loss))
                    if np.min(heights) < FLOOR_MIN_Z_M:
                        raise AgentError('Arm/gripper crosses floor minimum z %.6f m: %.6f m' %
                                         (FLOOR_MIN_Z_M, np.min(heights)))
                    # Refine ambiguous interior intervals to avoid slowing a high path
                    # merely because a coarse motion bound reaches the slow band.
                    nominal = float(np.min(heights-scene.floor_motion @ np.full(7, padding)))
                    floor_refine = floor_bound < FLOOR_MIN_Z_M + FLOOR_SLOW_BAND_M <= nominal and depth < 12
                # Subdivide derivative curves directly. Differentiating tiny
                # position subsegments amplifies cancellation by 1/span².
                bounded = (np.all(lower >= low_limit) and np.all(upper <= high_limit)
                           and np.max(np.abs(velocity)) <= settings.max_velocity+1e-6
                           and np.max(np.abs(acceleration)) <= settings.max_acceleration+1e-6)
                pair = worst if distances[worst] < .003 else int(np.argmin(bounds))
                violations = []
                for j, name in enumerate(JOINTS):
                    if lower[j] < low_limit[j] or upper[j] > high_limit[j]:
                        violations.append('%s position bound [%.9f, %.9f] rad outside '
                                          'allowed [%.9f, %.9f] rad' %
                                          (name, lower[j], upper[j], low_limit[j], high_limit[j]))
                    for label, values, limit, unit in (
                            ('velocity', velocity, settings.max_velocity, 'rad/s'),
                            ('acceleration', acceleration, settings.max_acceleration, 'rad/s²')):
                        peak = float(np.max(np.abs(values[:, j])))
                        if peak > limit+1e-6:
                            violations.append('%s %s bound %.9f > %.9f %s' %
                                              (name, label, peak, limit, unit))
                last_diagnostic = (
                    'waypoint interval %d (zero-based), time [%.9f, %.9f] s, depth %d; '
                    'pair %s / %s; midpoint nominal clearance %.9f m, interpolation deduction %.9f m, '
                    'tracking deduction %.9f m, clearance lower bound %.9f m, required 0.003000000 m; '
                    'tracking allowance %.6f rad per joint; motion-bound violations: %s '
                    '(bounds are conservative interpolation bounds, not measured hardware values)' %
                    (interval, at, at+span, depth, scene.names[scene.first[pair]],
                     scene.names[scene.second[pair]], distances[pair], interpolation_loss[pair],
                     tracking_loss[pair], bounds[pair], padding, '; '.join(violations) or 'none'))
                if settings.floor_guard:
                    last_diagnostic += '; floor clearance bound %.6f m (minimum z %.6f m)' % (floor_bound, FLOOR_MIN_Z_M)
                if distances[worst] < .003:
                    raise AgentError('MuJoCo clearance rejected: ' + last_diagnostic)
                if clearance >= .003 and bounded and floor_bound >= FLOOR_MIN_Z_M and not floor_refine:
                    minimum = min(minimum, clearance)
                    minimum_floor = min(minimum_floor, floor_bound)
                    continue
                if depth >= 16:
                    raise AgentError('MuJoCo could not certify interpolation bounds/clearance '
                                     'including %.3f-rad tracking allowance: ' % padding + last_diagnostic)
                vl, vr = split(velocity)
                al, ar = split(acceleration)
                pending.extend(((left, vl, al, span/2, depth+1, at),
                                (right, vr, ar, span/2, depth+1, at+span/2)))
        return {'status': 'passed', 'scope': 'kinematic geometry; not physical dynamics validation',
                'intervals_checked': count, 'minimum_clearance_bound_m': minimum,
                'tracking_allowance_rad': padding,
                'gripper_opening_allowance_m': GRIPPER_PREFLIGHT_ALLOWANCE_M,
                'minimum_floor_clearance_bound_m': minimum_floor if settings.floor_guard else None,
                'floor_minimum_z_m': FLOOR_MIN_Z_M if settings.floor_guard else None,
                'kinematic_inertial_placeholders': scene.inertial_placeholders,
                'robot_description_sha256': hashlib.sha256(description.encode()).hexdigest()}
    except AgentError:
        raise
    except Exception as error:
        raise AgentError('MuJoCo preflight unavailable or invalid: ' + str(error)) from error
