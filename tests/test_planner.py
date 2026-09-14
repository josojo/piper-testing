"""Offline acceptance tests. Run with unittest discover -s tests, not the hardware scripts."""

import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from nero_planner import Limits, Planner, PlanningError, Pose
from nero_planner.model import JOINT_NAMES, OPEN_GRIPPER, box_clearances
from scripts.build_nero_scene import DEFAULT_MODEL, build


ROOT = Path(__file__).resolve().parents[1]
SELF_COLLISION = [-0.0398420451, -0.6842809682, -1.2728772034, 2.0932450110, -1.5405981323, -0.6762259732, 1.1143164399]
TABLE_COLLISION = [-2.6214222195, -1.2156001735, -2.3420722116, 1.1211550686, 2.1528053850, -0.5931009774, -0.3257220589]
APPLE_COLLISION = [0.1954862676, -1.6571597954, -1.4925102770, 0.2731197333, -2.4038274022, 0.4873003702, 0.3809628509]
NEAR_MISS = [-1.7810100898, 1.1949029404, 1.2278108248, -0.7605014230, -2.1799482436, 0.0308864323, 0.7631764073]
# Both ends clear, but a table collision lies in the middle of this long path.
CROSSING_A = [0.9601933559, 1.1303715051, 1.1284510392, 1.6830510274, -1.6836514945, 0.6442726940, -0.5983538809]
CROSSING_B = [2.0047653309, 1.3708907963, -2.5736990452, 1.5785738878, 0.7106986002, 0.6382657659, 1.0266188106]
# A <= 0.25 rad move also violates clearance only between its endpoints.
SHORT_A = [-2.3083677293, -1.3447334401, 0.7123389920, 1.1099401865, 1.7190649879, 0.3768256853, -0.6434436252]
SHORT_B = [-2.5309691200, -1.3749729886, 0.5579496025, 1.2266675339, 1.5980479751, 0.4118498247, -0.6481850793]


class PlannerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not DEFAULT_MODEL.exists():
            raise RuntimeError("Prepare the offline model first: python scripts/prepare_nero_mujoco.py")
        cls.temp = tempfile.TemporaryDirectory()
        cls.scene_path = Path(cls.temp.name) / "scene.xml"
        with contextlib.redirect_stdout(io.StringIO()):
            build(DEFAULT_MODEL, cls.scene_path)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def setUp(self):
        self.planner = Planner(self.scene_path)
        self.planner.set_start([0.0] * 7)

    def target(self):
        pose = self.planner.current_pose()
        return replace(pose, position_m=(pose.position_m[0] + 0.02, pose.position_m[1], pose.position_m[2] - 0.002))

    def test_scene_mapping_tool_frame_and_gripper(self):
        scene = self.planner.scene
        self.assertEqual(scene.model.nq, 10)
        self.assertEqual(tuple(scene.model.joint(i).name for i in range(7)), JOINT_NAMES)
        self.assertEqual(scene.model.neq, 2)
        for name, value in OPEN_GRIPPER.items():
            self.assertAlmostEqual(scene.data.qpos[scene.model.joint(name).qposadr[0]], value)
        np.testing.assert_allclose(self.planner.current_pose().position_m, [0, 0, 0.89301], atol=1e-6)
        np.testing.assert_allclose(scene.model.site("grasp_center").quat,
                                   scene.model.body("gripper_link").quat, atol=1e-6)
        np.testing.assert_allclose(scene.model.site("grasp_center").pos,
                                   scene.model.body("gripper_link1").pos, atol=1e-6)
        self.assertEqual(scene.model.body("apple").jntnum[0], 0)

    def test_proxies_enclose_every_mesh_at_multiple_poses(self):
        scene = self.planner.scene
        m, d = scene.model, scene.data
        meshes = [i for i in range(m.ngeom) if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH]
        for q in (np.zeros(7), np.asarray(SELF_COLLISION), np.asarray(TABLE_COLLISION)):
            scene.set_arm(q)
            for source_index, gid in enumerate(meshes):
                body = m.body(m.geom_bodyid[gid]).name
                proxy = m.geom(f"collision_{body}_{source_index}").id
                mid = m.geom_dataid[gid]
                start = m.mesh_vertadr[mid]
                vertices = m.mesh_vert[start:start + m.mesh_vertnum[mid]]
                world = vertices @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
                local = (world - d.geom_xpos[proxy]) @ d.geom_xmat[proxy].reshape(3, 3)
                self.assertTrue(np.all(np.abs(local) <= m.geom_size[proxy] + 1e-6), body)

    def test_reach_and_playback(self):
        start = self.planner.scene.data.qpos.copy()
        trajectory = self.planner.plan(self.target())
        np.testing.assert_array_equal(self.planner.scene.data.qpos, start)
        report = trajectory.validation
        self.assertGreaterEqual(report.minimum_clearance_m, 0.03)
        self.assertLessEqual(report.position_error_m, 0.002)
        self.assertLessEqual(report.orientation_error_rad, np.deg2rad(2))
        self.assertLessEqual(report.peak_velocity_rad_s, 0.2 + 1e-12)
        self.assertLessEqual(report.peak_acceleration_rad_s2, 0.5 + 1e-12)
        self.assertTrue(np.all(np.diff(trajectory.timestamps) > 0))
        self.assertLessEqual(np.max(np.abs(np.asarray(trajectory.positions)[-1])), 0.25)
        self.planner.playback(trajectory)
        np.testing.assert_allclose(self.planner.scene.data.qpos[self.planner.scene.qadr], trajectory.positions[-1])
        # Independent dense sampling checks the certified lower bound.
        for alpha in np.linspace(0, 1, 201):
            q = (1 - alpha) * np.asarray(trajectory.positions[0]) + alpha * np.asarray(trajectory.positions[-1])
            self.assertGreaterEqual(self.planner.scene.distances(q).min(), report.minimum_clearance_m - 1e-9)

    def test_rejects_table_apple_and_self(self):
        for q, expected in ((TABLE_COLLISION, "table"), (APPLE_COLLISION, "apple"), (SELF_COLLISION, "collision_link2")):
            with self.subTest(expected=expected):
                self.planner.set_start(q)
                target = self.planner.current_pose()
                saved = self.planner.scene.data.qpos.copy()
                with self.assertRaisesRegex(PlanningError, expected):
                    self.planner.plan(target)
                np.testing.assert_array_equal(self.planner.scene.data.qpos, saved)

    def test_near_miss_without_forbidden_contact(self):
        scene = self.planner.scene
        distances = scene.distances(NEAR_MISS)
        self.assertGreater(distances.min(), 0)
        forbidden = {frozenset(pair) for pair in scene.pairs}
        self.assertFalse(any(frozenset((contact.geom1, contact.geom2)) in forbidden
                             for contact in scene.data.contact[:scene.data.ncon]))
        with self.assertRaisesRegex(PlanningError, "clearance"):
            self.planner._clearance(NEAR_MISS)

    def test_safe_endpoints_do_not_certify_middle(self):
        for a, b in ((CROSSING_A, CROSSING_B), (SHORT_A, SHORT_B)):
            with self.subTest(a=a):
                self.planner._clearance(a)
                self.planner._clearance(b)
                with self.assertRaisesRegex(PlanningError, "clearance"):
                    self.planner._certify_segment(np.asarray(a), np.asarray(b))
        self.assertLess(self.planner.scene.distances((np.asarray(CROSSING_A) + CROSSING_B) / 2).min(), 0)
        self.assertLessEqual(np.max(np.abs(np.asarray(SHORT_A) - SHORT_B)), 0.25)

    def test_unresolved_intervals_fail_closed(self):
        self.planner.limits = replace(Limits(), max_subdivision_depth=1)
        with self.assertRaisesRegex(PlanningError, "between samples"):
            self.planner.plan(self.target())
        self.planner.limits = replace(Limits(), max_validation_samples=3)
        with self.assertRaisesRegex(PlanningError, "budget"):
            self.planner.plan(self.target())

    def test_bad_targets(self):
        valid = {"frame": "nero_base", "position_m": [0, 0, 0.8], "orientation_xyzw": [0, 0, 0, 1]}
        bad = [None, {}, {**valid, "extra": 1}, {**valid, "frame": "camera"},
               {**valid, "position_m": [0, 0]}, {**valid, "position_m": [0, float("nan"), 0]},
               {**valid, "position_m": [True, 0, 0]}, {**valid, "position_m": ["0", 0, 0]},
               {**valid, "orientation_xyzw": [0, 0, 0, 0]}, {**valid, "orientation_xyzw": [0, 0, 0, 2]},
               {**valid, "gripper": 0}, {**valid, "gripper": True}, {**valid, "reason": 4}]
        for value in bad:
            with self.subTest(value=value), self.assertRaises(PlanningError):
                Pose.from_dict(value)

    def test_quaternion_sign_equivalence(self):
        target = self.target()
        a = self.planner.plan(target)
        b = self.planner.plan(replace(target, orientation_xyzw=tuple(-v for v in target.orientation_xyzw)))
        np.testing.assert_allclose(a.positions[-1], b.positions[-1], atol=1e-10)

    def test_noop_is_valid(self):
        trajectory = self.planner.plan(self.planner.current_pose())
        self.planner.playback(trajectory)
        self.assertEqual(trajectory.validation.peak_velocity_rad_s, 0)
        self.assertEqual(trajectory.validation.peak_acceleration_rad_s2, 0)

    def test_old_scene_reports_preparation_commands(self):
        tree = ET.parse(self.scene_path)
        root = tree.getroot()
        root.remove(root.find("equality"))
        actuators = root.find("actuator")
        actuators.remove(actuators.find("position[@joint='gripper']"))
        body = root.find(".//body[@name='gripper_link']")
        body.remove(body.find("joint[@name='gripper']"))
        path = Path(self.temp.name) / "old_scene.xml"
        tree.write(path)
        with self.assertRaises(ValueError) as caught:
            Planner(path)
        message = str(caught.exception)
        self.assertIn("missing: gripper", message)
        self.assertIn(str(path.resolve()), message)
        self.assertIn("prepare_nero_mujoco.py", message)
        self.assertIn("build_nero_scene.py", message)
        # An incomplete input must not overwrite an existing generated scene.
        output = Path(self.temp.name) / "preserved_scene.xml"
        output.write_text("existing scene")
        with self.assertRaisesRegex(SystemExit, "missing required joints: gripper"):
            build(path, output)
        self.assertEqual(output.read_text(), "existing scene")

    def test_extra_fixed_obstacles_and_unsupported_geometry(self):
        tree = ET.parse(self.scene_path)
        obstacle = ET.SubElement(tree.getroot().find("worldbody"), "geom", {
            "name": "extra_obstacle", "type": "box", "pos": "0 0 0.89301", "size": "0.08 0.08 0.08",
        })
        path = Path(self.temp.name) / "obstacle_scene.xml"
        tree.write(path)
        planner = Planner(path)
        with self.assertRaisesRegex(PlanningError, "extra_obstacle"):
            planner.plan(planner.current_pose())
        obstacle.set("type", "sphere")
        tree.write(path)
        with self.assertRaisesRegex(ValueError, "must be boxes"):
            Planner(path)

    def test_joint_mapping_with_interleaved_gripper_address(self):
        tree = ET.parse(self.scene_path)
        world = tree.getroot().find("worldbody")
        parent = world.find(".//body[@name='link7']")
        dummy = parent.find("body[@name='gripper_link']")
        # The vendor's dummy driver has no geometry. Reorder it before the
        # robot to make all seven arm qpos addresses shift by one.
        parent.remove(dummy)
        world.insert(0, dummy)
        path = Path(self.temp.name) / "reordered_scene.xml"
        tree.write(path)
        planner = Planner(path)
        self.assertEqual(planner.scene.qadr.tolist(), list(range(1, 8)))
        planner.set_start([0] * 7)
        trajectory = planner.plan(self.target())
        planner.playback(trajectory)
        self.assertAlmostEqual(planner.scene.data.qpos[0], 0.1)

    def test_translation_unreachable_orientation_and_joint_limits(self):
        pose = self.planner.current_pose()
        with self.assertRaisesRegex(PlanningError, "step limit"):
            self.planner.plan(replace(pose, position_m=(1, 0, 1)))
        with self.assertRaisesRegex(PlanningError, "IK failed"):
            self.planner.plan(replace(pose, position_m=(0, 0, pose.position_m[2] + 0.04)))
        with self.assertRaisesRegex(PlanningError, "IK failed"):
            self.planner.plan(replace(pose, orientation_xyzw=(1, 0, 0, 0)))
        for q in ([4, 0, 0, 0, 0, 0, 0], [float("nan")] * 7, [0] * 6):
            with self.assertRaises(PlanningError):
                self.planner.set_start(q)

    def test_duration_and_motion_limits(self):
        target = self.target()
        self.planner.limits = replace(Limits(), max_duration_s=0.01)
        with self.assertRaisesRegex(PlanningError, "duration"):
            self.planner.plan(target)
        self.planner.limits = replace(Limits(), max_joint_displacement_rad=0.001)
        with self.assertRaisesRegex(PlanningError, "IK failed"):
            self.planner.plan(target)
        for kwargs in ({"minimum_clearance_m": -1}, {"max_velocity_rad_s": float("inf")},
                       {"max_ik_iterations": 1.5}, {"sample_period_s": 0}):
            with self.assertRaises(PlanningError):
                Limits(**kwargs)

    def test_stale_state_scene_limits_and_tampering(self):
        trajectory = self.planner.plan(self.target())
        self.planner.set_start([0.01, 0, 0, 0, 0, 0, 0])
        with self.assertRaisesRegex(PlanningError, "starting state changed"):
            self.planner.playback(trajectory)
        self.planner.set_start([0] * 7)
        with self.assertRaisesRegex(PlanningError, "modified"):
            self.planner.playback(replace(trajectory, timestamps=(0, 0.1)))
        self.planner.limits = replace(Limits(), max_velocity_rad_s=0.1)
        with self.assertRaisesRegex(PlanningError, "limits changed"):
            self.planner.playback(trajectory)
        self.planner.limits = Limits()
        self.planner.scene.model.body("apple").pos[0] += 0.1
        with self.assertRaisesRegex(PlanningError, "Scene or limits changed"):
            self.planner.playback(trajectory)
        with self.assertRaisesRegex(PlanningError, "Scene was modified"):
            self.planner.plan(trajectory.source_target_pose)

    def test_nonstationary_and_unsynchronized_start(self):
        target = self.target()
        self.planner.scene.data.qvel[0] = 0.01
        self.planner.current_pose()
        with self.assertRaisesRegex(PlanningError, "stationary"):
            self.planner.plan(target)
        self.planner.scene.data.qvel[:] = 0
        self.planner.scene.data.qpos[self.planner.scene.model.joint("gripper").qposadr[0]] = 0
        self.planner.current_pose()
        with self.assertRaisesRegex(PlanningError, "gripper"):
            self.planner.plan(target)

    def test_cli_headless_and_rejection_exit_code(self):
        output = Path(self.temp.name) / "trajectory.json"
        result = subprocess.run([sys.executable, "-m", "nero_planner", "--scene", str(self.scene_path),
                                 "--target", str(ROOT / "examples/reach.json"), "--output", str(output)],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "reached")
        self.assertEqual(json.loads(output.read_text())["trajectory"]["joint_names"], list(JOINT_NAMES))
        bad = Path(self.temp.name) / "bad.json"
        bad.write_text('{"frame":"camera"}')
        result = subprocess.run([sys.executable, "-m", "nero_planner", "--scene", str(self.scene_path),
                                 "--target", str(bad)], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(json.loads(result.stderr)["status"], "rejected")
        self.assertFalse(any(name.startswith("pyAgxArm") for name in sys.modules))


class DistanceTests(unittest.TestCase):
    def test_separation_overlap_and_rotation(self):
        sizes = np.ones((2, 3))
        rotations = np.repeat(np.eye(3)[None], 2, axis=0)
        centers = np.array([[0., 0, 0], [3., 0, 0]])
        def distance():
            return box_clearances(centers, rotations, sizes, np.array([0]), np.array([1]))[0]
        self.assertAlmostEqual(distance(), 1, places=8)
        centers[1, 0] = 1.5
        self.assertLess(distance(), 0)
        theta = np.pi / 4
        rotations[1] = [[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]]
        centers[1, 0] = 3
        self.assertAlmostEqual(distance(), 2 - np.sqrt(2), places=8)
        # A diagonal separation is conservatively smaller than true distance.
        rotations[1] = np.eye(3)
        centers[1] = [3, 3, 0]
        self.assertLessEqual(distance(), np.sqrt(2))


if __name__ == "__main__":
    unittest.main()
