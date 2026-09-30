"""Browser playback of portable candidate previews; no ROS or hardware connection."""
import argparse
from pathlib import Path
import re
import time
import queue
import threading
from datetime import datetime

import mujoco
import numpy as np

from .core import JOINTS
from .preview import load_preview, sample


def rotation_quaternion(matrix):
    quaternion = np.empty(4)
    mujoco.mju_mat2Quat(quaternion, np.asarray(matrix, dtype=float).reshape(9))
    return quaternion


def mesh_arrays(model, geom_id):
    """MuJoCo's compiled mesh coordinates match geom_xpos/geom_xmat."""
    mesh_id = model.geom_dataid[geom_id]
    start = model.mesh_vertadr[mesh_id]
    face_start = model.mesh_faceadr[mesh_id]
    return (model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]].copy(),
            model.mesh_face[face_start:face_start + model.mesh_facenum[mesh_id]].copy())


def create_viewer(path, host='127.0.0.1', port=8080, speed=.25, *, server=None, loaded=None):
    import viser

    payload, model, data = loaded if loaded is not None else load_preview(path)
    if payload['joint_names'] != list(JOINTS):
        raise ValueError('Preview joint order does not match the NERO model')
    qadr = [model.jnt_qposadr[model.joint(name).id] for name in JOINTS]
    geom_ids = [model.geom(name).id for name in payload['geom_names']]
    sizes, offsets = np.asarray(payload['sizes']), np.asarray(payload['offsets'])
    start, end = payload['times'][0], payload['times'][-1]
    server = server or viser.ViserServer(host=host, port=port)
    server.scene.set_up_direction('+z')
    server.initial_camera.position = (1.4, -1.4, 1.1)
    server.initial_camera.look_at = (0., 0., .35)
    server.gui.add_markdown('## Saved trajectory preview\nKinematic playback of the exported collision model. No hardware connection.')
    server.gui.add_text('Validation status', initial_value=payload['validation_status'], disabled=True)
    if payload.get('reason'):
        server.gui.add_text('Diagnostic', initial_value=payload['reason'], multiline=True, disabled=True)
    playing = server.gui.add_checkbox('Play', initial_value=False)
    timeline = server.gui.add_slider('Time (s)', min=start, max=end, step=min(.01, (end-start)/1000), initial_value=start)
    speed_control = server.gui.add_number('Speed', initial_value=speed, min=.01, max=10., step=.05)
    boxes = server.gui.add_checkbox('Padded collision boxes', initial_value=False)
    reset = server.gui.add_button('Restart')

    @reset.on_click
    def reset_time(_):
        playing.value = False
        timeline.value = start

    failure = re.search(r'time \[([0-9.]+), ([0-9.]+)\]', payload.get('reason') or '')
    if failure:
        jump = server.gui.add_button('Jump to rejected interval')

        @jump.on_click
        def jump_time(_):
            playing.value = False
            timeline.value = float(np.clip((float(failure[1])+float(failure[2]))/2, start, end))

    pair = re.search(r'pair (\S+) / (\S+);', payload.get('reason') or '')
    highlighted = set(pair.groups()) if pair else set()
    handles, bounds = [], []
    for i, gid in enumerate(geom_ids):
        name = payload['geom_names'][i]
        color = (230, 50, 35) if name in highlighted else (165, 180, 200)
        if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
            vertices, faces = mesh_arrays(model, gid)
            handle = server.scene.add_mesh_simple('/robot/'+name, vertices, faces, color=color)
        elif model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_BOX:
            handle = server.scene.add_box('/robot/'+name, dimensions=tuple(2*model.geom_size[gid]), color=color)
        else:
            server.stop()
            raise ValueError('Unsupported preview geometry: '+name)
        handles.append(handle)
        bounds.append(server.scene.add_box('/bounds/'+name, dimensions=tuple(2*sizes[i]),
                                          color=(50, 140, 255), opacity=.18, visible=False))
    for i, (center, rot, name) in enumerate(zip(payload['obstacles'], payload['obstacle_rotations'], payload['obstacle_names'])):
        server.scene.add_box('/obstacles/'+name, dimensions=tuple(2*sizes[len(geom_ids)+i]),
                             position=tuple(center), wxyz=rotation_quaternion(rot),
                             color=(230, 50, 35) if name in highlighted else (130, 130, 130), opacity=.6)
    if payload.get('floor_minimum_z_m') is not None:
        transform = np.linalg.inv(payload['base_from_root'])
        center = transform @ [0, 0, payload['floor_minimum_z_m'], 1]
        server.scene.add_box('/floor', dimensions=(2., 2., .001), position=tuple(center[:3]),
                             wxyz=rotation_quaternion(transform[:3, :3]), color=(60, 180, 80), opacity=.15)
    trace_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'preflight_gripper_base_0')
    if trace_id >= 0:
        trail = []
        for at in np.linspace(start, end, 200):
            data.qpos[qadr] = sample(payload, at)
            mujoco.mj_forward(model, data)
            trail.append(data.geom_xpos[trace_id].copy())
        trail = np.asarray(trail)
        server.scene.add_line_segments('/gripper-base-path', points=np.stack((trail[:-1], trail[1:]), axis=1),
                                       colors=(255, 185, 25), thickness=3., thickness_units='screen')
        server.gui.add_markdown('Yellow line: gripper-base center. Blue boxes: nominal padded geometry, not a tracking-error envelope.')

    def update():
        data.qpos[qadr] = sample(payload, timeline.value)
        mujoco.mj_forward(model, data)
        with server.atomic():
            for i, gid in enumerate(geom_ids):
                rot = data.geom_xmat[gid].reshape(3, 3)
                quat = rotation_quaternion(rot)
                handles[i].position = data.geom_xpos[gid].copy()
                handles[i].wxyz = quat
                bounds[i].position = data.geom_xpos[gid] + rot @ offsets[i]
                bounds[i].wxyz = quat
                bounds[i].visible = boxes.value

    update()
    return server, update, playing, timeline, speed_control



def add_planning_controls(server, planner, completed):
    """Run planning off the render thread; only that thread replaces the scene."""
    server.gui.add_markdown('## New TCP target\nPosition in base_link, meters. Quaternion order: X/Y/Z/W. '
                            'Plans from fresh feedback; keep the robot stationary. Calculation does not execute motion.')
    defaults = next((p for p in planner.defaults['named_poses'].values() if 'position_m' in p),
                    {'position_m': [0.3, 0., 0.4], 'orientation_xyzw': [0., 0., 0., 1.]})
    position = server.gui.add_vector3('Target XYZ (m)', initial_value=tuple(defaults['position_m']), step=.005)
    orientation = [server.gui.add_number('Quaternion '+axis, initial_value=float(value), step=.01)
                   for axis, value in zip('XYZW', defaults['orientation_xyzw'])]
    status = server.gui.add_text('Planning result', initial_value='Ready', disabled=True, multiline=True)
    button = server.gui.add_button('Calculate trajectory')
    busy = threading.Lock()

    @button.on_click
    def calculate(_):
        if not busy.acquire(blocking=False):
            return
        xyz = tuple(position.value)
        xyzw = tuple(item.value for item in orientation)
        button.disabled = True
        status.value = 'Calculating trajectory and checking collisions…'

        def work():
            try:
                candidate = planner(xyz, xyzw)
                load_preview(candidate)
                completed.put(candidate)
            except Exception as error:
                status.value = str(error)
            finally:
                button.disabled = False
                busy.release()
        threading.Thread(target=work, daemon=True).start()
    return button, status


class PreviewSource:
    """Watch a selected file, retaining the displayed revision on read errors."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.revision = None

    def read(self, force=False):
        stat = self.path.stat()
        revision = (stat.st_ino, stat.st_mtime_ns, stat.st_size)
        if not force and revision == self.revision:
            return None
        loaded = load_preview(self.path)
        payload, model, _ = loaded
        if payload['joint_names'] != list(JOINTS):
            raise ValueError('Preview joint order does not match the NERO model')
        for name in payload['geom_names']:
            gid = model.geom(name).id
            if model.geom_type[gid] not in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_BOX):
                raise ValueError('Unsupported preview geometry: ' + name)
        after = self.path.stat()
        if revision != (after.st_ino, after.st_mtime_ns, after.st_size):
            raise ValueError('Preview is being replaced; retrying shortly')
        return loaded, revision


class LivePreview:
    def __init__(self, server, path, speed=.25):
        self.server = server
        self.source = PreviewSource(path)
        self.requests = queue.Queue()
        self.folder = None
        self.speed_value = speed
        self.next_check = 0.
        self.path_input = server.gui.add_text('Preview file', initial_value=str(self.source.path))
        self.auto = server.gui.add_checkbox('Auto-reload selected file', initial_value=True)
        self.status = server.gui.add_text('File status', initial_value='', disabled=True, multiline=True)
        self.loaded_file = server.gui.add_text('Displayed file', initial_value='', disabled=True)
        self.loaded_time = server.gui.add_text('Loaded at', initial_value='', disabled=True)
        button = server.gui.add_button('Load / reload file')

        @button.on_click
        def reload_file(_):
            self.requests.put(Path(self.path_input.value).expanduser().resolve())

        self.reload(force=True)

    def reload(self, path=None, force=False):
        source = PreviewSource(path) if path is not None else self.source
        try:
            result = source.read(force=force)
            if result is None:
                self.status.value = 'Watching for changes' if self.auto.value else 'Auto-reload off'
                return
            loaded, revision = result
        except Exception as error:
            self.status.value = 'Could not load %s: %s. Keeping displayed trajectory.' % (source.path, error)
            return
        if self.folder is not None:
            self.playing.value = False
            self.speed_value = self.speed.value
            self.folder.remove()
            self.server.scene.reset()
        self.folder = self.server.gui.add_folder('Trajectory playback')
        with self.folder:
            _, self.update, self.playing, self.timeline, self.speed = create_viewer(
                source.path, speed=self.speed_value, server=self.server, loaded=loaded)
        source.revision = revision
        self.source = source
        self.path_input.value = str(source.path)
        self.loaded_file.value = str(source.path)
        self.loaded_time.value = datetime.now().astimezone().isoformat(timespec='seconds')
        self.status.value = 'Loaded newest file; playback paused'

    def tick(self, now):
        try:
            path = self.requests.get_nowait()
        except queue.Empty:
            path = None
        if path is not None:
            self.reload(path, force=True)
            self.next_check = now + 1.
        elif now >= self.next_check:
            if self.auto.value:
                self.reload()
            self.next_check = now + 1.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('preview', type=Path)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    parser.add_argument('--speed', type=float, default=.25)
    parser.add_argument('--planning-config', type=Path, help='Enable target planning against an existing ROS stack')
    parser.add_argument('--ros-container', help='Existing ROS Docker container ID/name (requires Docker access)')
    args = parser.parse_args()
    if args.ros_container and not args.planning_config:
        parser.error('--ros-container requires --planning-config')
    if not np.isfinite(args.speed) or not .01 <= args.speed <= 10:
        parser.error('--speed must be between 0.01 and 10')
    import viser
    # Fail clearly if the initial preview cannot be read.
    load_preview(args.preview)
    server = viser.ViserServer(host=args.host, port=args.port)
    live = LivePreview(server, args.preview, args.speed)
    completed = queue.Queue()
    planner = None
    if args.planning_config:
        from .viewer_planning import ViewerPlanner
        planner = ViewerPlanner(args.planning_config, args.ros_container)
        add_planning_controls(server, planner, completed)
    print(f'Open http://localhost:{args.port} (forward this port if accessing over SSH). Starts paused.', flush=True)
    last = time.monotonic()
    try:
        while True:
            if not completed.empty():
                live.requests.put(completed.get_nowait())
            now = time.monotonic()
            live.tick(now)
            if live.playing.value:
                live.timeline.value = min(live.timeline.max, live.timeline.value + (now-last)*live.speed.value)
                if live.timeline.value >= live.timeline.max:
                    live.playing.value = False
            last = now
            live.update()
            time.sleep(1/30)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == '__main__':
    main()
