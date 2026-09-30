"""Portable offline exports of exact controller curves, including rejections."""
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from nero_agent.core import AgentError, JOINTS
from nero_agent.mujoco_preflight import CollisionScene
from nero_agent.preview import export_preview, load_preview, sample
from nero_agent.ros_backend import RosBackend
from tests.test_mujoco_preflight import description, config, point


class PreviewTests(unittest.TestCase):
    def test_portable_model_and_spline_round_trip(self):
        q = [.6, 1.236, -.034, .729, .037, -.007, -.222]
        goal = q.copy(); goal[0] += .1
        first, last = point(q, 0), point(goal, 2)
        first.velocities[0] = .02
        t = NS(joint_names=list(JOINTS), points=[first, last])
        before = {'gripper_width_m': .04}
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp)/'source'; source.mkdir()
            path = source/'preview.json'
            export_preview(path, description(), t, before, config(), 'rejected', 'test rejection')
            target = Path(temp)/'relocated'
            shutil.copytree(source, target)
            shutil.rmtree(source)
            payload, model, data = load_preview(target/'preview.json')
            self.assertEqual(payload['validation_status'], 'rejected')
            self.assertEqual(payload['reason'], 'test rejection')
            np.testing.assert_allclose(sample(payload, 0), q)
            np.testing.assert_allclose(sample(payload, 2), goal)
            self.assertGreater(sample(payload, 1)[0], (q[0]+goal[0])/2)
            scene = CollisionScene(description(), config().collision_boxes, .04)
            for at in (0, .5, 1, 2):
                position = sample(payload, at)
                scene.distances(position)
                data.qpos[[model.jnt_qposadr[model.joint(n).id] for n in JOINTS]] = position
                mujoco.mj_forward(model, data)
                for name in payload['geom_names']:
                    np.testing.assert_allclose(data.geom_xpos[model.geom(name).id],
                                               scene.data.geom_xpos[scene.model.geom(name).id], atol=2e-6)

    def test_rejected_backend_still_exports_and_still_raises(self):
        backend = RosBackend.__new__(RosBackend)
        backend.settings = config()
        q = [0.]*7
        t = NS(joint_names=list(JOINTS), points=[point(q, 0), point(q, 1)])
        with tempfile.TemporaryDirectory() as temp:
            backend.preview_output = Path(temp)/'rejected.json'
            with patch.object(backend, '_check_mujoco_preflight', side_effect=AgentError('clearance failed')):
                with self.assertRaisesRegex(AgentError, 'clearance failed'):
                    backend._mujoco_preflight(description(), NS(joint_trajectory=t), {'gripper_width_m': .04})
            payload, _, _ = load_preview(backend.preview_output)
            self.assertEqual(payload['validation_status'], 'rejected')
            self.assertEqual(payload['reason'], 'clearance failed')

    def test_passed_export_contains_final_retimed_curve(self):
        backend = RosBackend.__new__(RosBackend)
        backend.settings = config()
        q = [0.]*7
        t = NS(joint_names=list(JOINTS), points=[point(q, 0), point(q, 1)])
        def check(*args):
            t.points[-1].time_from_start.sec = 5
            return {'mujoco_preflight': {'status': 'passed'}}
        with tempfile.TemporaryDirectory() as temp:
            backend.preview_output = Path(temp)/'passed.json'
            with patch.object(backend, '_check_mujoco_preflight', side_effect=check):
                backend._mujoco_preflight(description(), NS(joint_trajectory=t), {'gripper_width_m': .04})
            payload, _, _ = load_preview(backend.preview_output)
            self.assertEqual(payload['times'], [0., 5.])
            self.assertEqual(payload['validation_status'], 'passed_geometric_preflight')
