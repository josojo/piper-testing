"""Offline MuJoCo inspection of an exported candidate; never commands hardware."""
import argparse
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .core import JOINTS, FLOOR_MIN_Z_M
from .mujoco_preflight import CollisionScene, bezier_segment


def seconds(point):
    return point.time_from_start.sec + point.time_from_start.nanosec * 1e-9


def export_preview(path, description, trajectory, before, settings, status, reason=None):
    """Save actual controller splines plus a portable model, even for rejected paths."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    scene = CollisionScene(description, settings.collision_boxes, before['gripper_width_m'])
    assets = Path(tempfile.mkdtemp(prefix=path.stem + '-', dir=path.parent))
    model_path = assets / 'robot.xml'
    mujoco.mj_saveLastXML(str(model_path), scene.model)
    root = ET.parse(model_path).getroot()
    # Copy assets out of Docker/vendor paths so the host viewer needs no ROS.
    for index, mesh in enumerate(root.findall('./asset/mesh')):
        source = Path(mesh.get('file'))
        destination = assets / ('mesh_%d%s' % (index, source.suffix))
        shutil.copyfile(source, destination)
        mesh.set('file', destination.name)
    ET.ElementTree(root).write(model_path, encoding='unicode')
    points = trajectory.points
    payload = {
        'format': 'nero-candidate-preview-v1', 'validation_status': status,
        'reason': reason, 'scope': 'Kinematic candidate preview only; not execution authorization',
        'mujoco_version': mujoco.__version__, 'model_file': str(model_path.relative_to(path.parent)),
        'joint_names': list(JOINTS), 'qpos': scene.data.qpos.tolist(),
        'times': [seconds(p) for p in points],
        'curves': [bezier_segment(a, b, seconds(b)-seconds(a)).tolist()
                   for a, b in zip(points, points[1:])],
        'geom_names': scene.names[:scene.model.ngeom],
        'offsets': scene.offsets.tolist(), 'sizes': scene.sizes.tolist(),
        'obstacles': scene.obstacles.tolist(), 'obstacle_rotations': scene.obstacle_rotations.tolist(),
        'obstacle_names': [b['id'] for b in settings.collision_boxes],
        'base_from_root': scene.base_from_root.tolist(),
        'floor_minimum_z_m': FLOOR_MIN_Z_M if settings.floor_guard else None,
    }
    # Publish only after the model/assets and complete JSON are ready.
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, prefix=path.name + '.',
                                         suffix='.tmp', delete=False) as output:
            temporary = Path(output.name)
            output.write(json.dumps(payload, indent=2, allow_nan=False)+'\n')
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return str(path)


def sample(payload, at):
    times = np.asarray(payload['times'])
    at = np.clip(at, times[0], times[-1])
    index = min(max(int(np.searchsorted(times, at, side='right'))-1, 0), len(times)-2)
    u = (at-times[index])/(times[index+1]-times[index])
    curve = np.asarray(payload['curves'][index], dtype=float)
    # de Casteljau evaluates the same quintic as controller/preflight.
    while len(curve) > 1:
        curve = (1-u)*curve[:-1] + u*curve[1:]
    return curve[0]


def load_preview(path):
    path = Path(path)
    payload = json.loads(path.read_text())
    if payload.get('format') != 'nero-candidate-preview-v1':
        raise ValueError('Expected an exported candidate preview, not the summary report')
    times, curves = np.asarray(payload['times']), np.asarray(payload['curves'])
    if (len(times) < 2 or not np.isfinite(times).all() or np.any(np.diff(times) <= 0)
            or curves.shape != (len(times)-1, 6, 7) or not np.isfinite(curves).all()):
        raise ValueError('Invalid preview trajectory')
    model = mujoco.MjModel.from_xml_path(str(path.parent / payload['model_file']))
    data = mujoco.MjData(model)
    data.qpos[:] = payload['qpos']
    return payload, model, data


def view(path, speed=.25):
    import mujoco.viewer
    payload, model, data = load_preview(path)
    qadr = [model.jnt_qposadr[model.joint(name).id] for name in JOINTS]
    geom_ids = [model.geom(name).id for name in payload['geom_names']]
    offsets, sizes = np.asarray(payload['offsets']), np.asarray(payload['sizes'])
    duration = payload['times'][-1]
    control = {'at': 0., 'playing': False, 'boxes': True}
    failure = re.search(r'time \[([0-9.]+), ([0-9.]+)\]', payload.get('reason') or '')
    failure_time = (float(failure[1])+float(failure[2]))/2 if failure else 0.
    def key(code):
        if code == 32: control['playing'] = not control['playing']  # Space
        elif code in (262, 263):  # arrows, 50 ms of trajectory time
            control['playing'] = False
            control['at'] = np.clip(control['at'] + (.05 if code == 262 else -.05), 0., duration)
        elif code == 82: control.update(at=0., playing=False)  # R
        elif code == 66: control['boxes'] = not control['boxes']  # B
        elif code == 74: control.update(at=failure_time, playing=False)  # J
    print('Candidate status:', payload['validation_status'])
    if payload['reason']: print(payload['reason'])
    print('OFFLINE ONLY. Starts paused. Space: play/pause; arrows: scrub 50 ms; R: reset; B: boxes; J: rejection time.')
    print('Playback speed: %gx. Red = diagnostic pair, blue = padded collision boxes, yellow = gripper-base trace.' % speed)
    print('Tracking uncertainty is in the diagnostic text; boxes show nominal geometry, not its envelope.')
    pair = re.search(r'pair (\S+) / (\S+);', payload.get('reason') or '')
    highlighted = set(pair.groups()) if pair else set()
    for name in highlighted:
        if name in payload['geom_names']:
            model.geom_rgba[model.geom(name).id] = [1, .15, .1, 1]
    trace_id = model.geom('preflight_gripper_base_0').id
    trail = []
    for at in np.linspace(0, duration, 100):
        data.qpos[qadr] = sample(payload, at)
        mujoco.mj_forward(model, data)
        trail.append(data.geom_xpos[trace_id].copy())
    data.qpos[qadr] = sample(payload, 0.)
    mujoco.mj_forward(model, data)
    with mujoco.viewer.launch_passive(model, data, key_callback=key) as window:
        window.cam.lookat[:] = np.mean(trail, axis=0)
        window.cam.distance = 1.6
        window.cam.azimuth, window.cam.elevation = 135, -25
        last = time.monotonic()
        while window.is_running():
            now = time.monotonic()
            if control['playing']:
                control['at'] = min(duration, control['at'] + (now-last)*speed)
                if control['at'] == duration: control['playing'] = False
            last = now
            with window.lock():
                data.qpos[qadr] = sample(payload, control['at'])
                data.time = control['at']
                mujoco.mj_forward(model, data)
                scn = window.user_scn
                scn.ngeom = 0
                def geom(kind, size, pos, rot, color):
                    g = scn.geoms[scn.ngeom]
                    mujoco.mjv_initGeom(g, kind, np.asarray(size, dtype=float), np.asarray(pos, dtype=float),
                                       np.asarray(rot, dtype=float).reshape(9), np.asarray(color, dtype=np.float32))
                    scn.ngeom += 1
                for p in trail:
                    geom(mujoco.mjtGeom.mjGEOM_SPHERE, [.002]*3, p, np.eye(3), [1, .7, .1, 1])
                for i, (center, rot, name) in enumerate(zip(payload['obstacles'], payload['obstacle_rotations'], payload['obstacle_names'])):
                    color = [1, .15, .1, .6] if name in highlighted else [.4, .4, .4, .5]
                    geom(mujoco.mjtGeom.mjGEOM_BOX, sizes[len(geom_ids)+i], center, rot, color)
                if payload['floor_minimum_z_m'] is not None:
                    root_from_base = np.linalg.inv(payload['base_from_root'])
                    center = root_from_base @ [0, 0, payload['floor_minimum_z_m'], 1]
                    geom(mujoco.mjtGeom.mjGEOM_BOX, [1., 1., .0005], center[:3],
                         root_from_base[:3, :3], [.2, .8, .2, .15])
                if control['boxes']:
                    for i, gid in enumerate(geom_ids):
                        rot = data.geom_xmat[gid].reshape(3, 3)
                        center = data.geom_xpos[gid] + rot @ offsets[i]
                        color = [1, .1, .1, .3] if payload['geom_names'][i] in highlighted else [.1, .5, 1, .18]
                        geom(mujoco.mjtGeom.mjGEOM_BOX, sizes[i], center, rot, color)
            window.sync()
            time.sleep(.01)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('preview', type=Path)
    parser.add_argument('--speed', type=float, default=.25)
    parser.add_argument('--check', action='store_true', help='Load and sample without a graphical display')
    args = parser.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        parser.error('--speed must be finite and positive')
    if args.check:
        payload, model, data = load_preview(args.preview)
        for at in payload['times']:
            data.qpos[[model.jnt_qposadr[model.joint(n).id] for n in JOINTS]] = sample(payload, at)
            mujoco.mj_forward(model, data)
        print(json.dumps({'preview': str(args.preview), 'status': payload['validation_status'],
                          'duration_s': payload['times'][-1], 'hardware_connected': False}))
    else:
        view(args.preview, args.speed)


if __name__ == '__main__':
    main()
