import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from nero_agent.viewer_planning import ViewerPlanner, target_config

ROOT = Path(__file__).resolve().parents[1]


class ViewerPlanningTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / 'examples/nero-agent.mock.json').read_text())
        self.config['motion_profile'] = 'mujoco_floor'

    def test_target_validation_preserves_limits_and_scene(self):
        result = target_config(self.config, [.3, 0, .4], [0, 0, 0, 1])
        self.assertEqual(result['collision_boxes'], self.config['collision_boxes'])
        self.assertEqual(result['motion_profile'], self.config['motion_profile'])
        self.assertNotIn('viewer_target', self.config['named_poses'])
        for xyz, quat in [([float('nan'), 0, 0], [0, 0, 0, 1]),
                          ([0, 0, 0], [0, 0, 0, 0])]:
            with self.assertRaises(Exception):
                target_config(self.config, xyz, quat)

    def test_container_request_is_plan_only_and_uses_unique_output(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'reports') as temp:
            directory = Path(temp)
            planner = ViewerPlanner(ROOT / 'examples/nero-agent.mock.json', 'ros-test')
            planner.defaults = self.config
            def run(command, **kwargs):
                self.assertEqual(command[:6], ['docker', 'exec', '-w', '/work', 'ros-test', '/nero_entrypoint.sh'])
                self.assertIn('nero_agent.viewer_planning', command)
                self.assertNotIn('--execute', command)
                self.assertNotIn('nero_agent.bringup', command)
                (directory / 'preview.json').write_text('{}')
                return SimpleNamespace(returncode=0)
            with patch('nero_agent.viewer_planning.tempfile.mkdtemp', return_value=temp), patch('nero_agent.viewer_planning.subprocess.run', side_effect=run):
                result = planner([.3, 0, .4], [0, 0, 0, 1])
            self.assertEqual(result, directory / 'preview.json')
            request = json.loads((directory / 'config.json').read_text())
            self.assertEqual(request['named_poses']['viewer_target']['position_m'], [.3, 0, .4])
