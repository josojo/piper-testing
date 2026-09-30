"""Exercise browser scene creation and spline scrubbing without robot access."""
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from urllib.request import urlopen

import mujoco
import numpy as np

from nero_agent.core import JOINTS
from nero_agent.preview import export_preview, load_preview, sample
from nero_agent.viser_preview import create_viewer, mesh_arrays, rotation_quaternion, LivePreview
from tests.test_mujoco_preflight import description, config, point


class ViserPreviewTests(unittest.TestCase):
    def test_live_reload_and_invalid_replacement(self):
        import viser
        q = [.6, 1.236, -.034, .729, .037, -.007, -.222]
        goal = q.copy()
        goal[0] += .1
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'preview.json'
            def export(duration):
                export_preview(path, description(), NS(points=[point(q, 0), point(goal, duration)]),
                               {'gripper_width_m': .04}, config(), 'test_fixture')
            export(2)
            server = viser.ViserServer(port=18081)
            try:
                live = LivePreview(server, path)
                live.playing.value = True
                live.timeline.value = 1
                export(4)
                live.tick(10)
                self.assertEqual(live.timeline.max, 4)
                self.assertEqual(live.timeline.value, 0)
                self.assertFalse(live.playing.value)
                revision = live.source.revision
                valid = path.read_text()
                path.write_text('{')
                live.tick(12)
                self.assertEqual(live.source.revision, revision)
                self.assertEqual(live.timeline.max, 4)
                self.assertIn('Keeping displayed trajectory', live.status.value)
                path.write_text(valid)
                live.tick(14)
                self.assertNotEqual(live.source.revision, revision)
                live.auto.value = False
                export(6)
                live.tick(16)
                self.assertEqual(live.timeline.max, 4)
                live.requests.put(path)
                live.tick(18)
                self.assertEqual(live.timeline.max, 6)
                other = Path(temp) / 'other.json'
                other.write_text(valid)
                live.requests.put(other)
                live.tick(20)
                self.assertEqual(live.timeline.max, 4)
                self.assertEqual(live.loaded_file.value, str(other))
                live.update()
            finally:
                server.stop()

    def test_scene_uses_exported_geometry_and_scrubbed_transforms(self):
        q = [.6, 1.236, -.034, .729, .037, -.007, -.222]
        goal = q.copy()
        goal[0] += .1
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'preview.json'
            export_preview(path, description(), NS(points=[point(q, 0), point(goal, 2)]),
                           {'gripper_width_m': .04}, config(), 'test_fixture')
            payload, model, data = load_preview(path)
            server, update, playing, timeline, speed = create_viewer(path, port=18080)
            try:
                with urlopen('http://127.0.0.1:%d' % server.get_port(), timeout=5) as response:
                    self.assertEqual(response.status, 200)
                    self.assertIn(b'<html', response.read().lower())
                timeline.value = 1.25
                update()
                data.qpos[[model.jnt_qposadr[model.joint(n).id] for n in JOINTS]] = sample(payload, 1.25)
                mujoco.mj_forward(model, data)
                for name in payload['geom_names']:
                    gid = model.geom(name).id
                    handle = server.scene._handle_from_node_name['/robot/'+name]
                    np.testing.assert_allclose(handle.position, data.geom_xpos[gid], atol=1e-7)
                    np.testing.assert_allclose(handle.wxyz, rotation_quaternion(data.geom_xmat[gid]), atol=1e-7)
                    if model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH:
                        vertices, faces = mesh_arrays(model, gid)
                        self.assertGreater(len(vertices), 0)
                        self.assertLess(faces.max(), len(vertices))
                self.assertFalse(playing.value)
                self.assertEqual(speed.value, .25)
            finally:
                server.stop()


if __name__ == '__main__':
    unittest.main()
