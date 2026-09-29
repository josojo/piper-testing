"""Offline geometry and backwards-reference checks for the shared TCP."""
from pathlib import Path
import struct
import tempfile
import unittest
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from nero_agent.tool_frame import (compose, from_rpy, migrate_flange_pose,
                                   rotate, tcp_in_link7, xacro_mappings)
from scripts.build_nero_scene import build, DEFAULT_MODEL


class ToolFrameTests(unittest.TestCase):
    def test_contact_surface_supports_model_default(self):
        # Independent evidence from the supplied original jaw mesh: its foremost
        # inner flat surface extends about 10 mm behind the 138 mm tip plane.
        blob = DEFAULT_MODEL.with_name('gripper_link1.stl').read_bytes()
        count = struct.unpack_from('<I', blob, 80)[0]
        dtype = np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')])
        faces = np.frombuffer(blob, dtype=dtype, offset=84, count=count)['v']
        contact = faces[np.max(np.abs(faces[:, :, 2]), axis=1) < 1e-7]
        self.assertGreater(len(contact), 0)
        depth = .138 + (contact[:, :, 1].min() + contact[:, :, 1].max())/2
        self.assertAlmostEqual(float(depth), .133, delta=.00002)

    def test_mujoco_matches_ros_offset_and_finger_midpoint(self):
        root = ET.parse(DEFAULT_MODEL).getroot()
        mappings = xacro_mappings(root)
        expected_p = np.array(list(map(float, mappings['tcp_offset_xyz'].split())))
        expected_q = from_rpy(list(map(float, mappings['tcp_offset_rpy'].split())))
        with tempfile.TemporaryDirectory() as tmp:
            scene = Path(tmp) / 'scene.xml'
            build(DEFAULT_MODEL, scene)
            m = mujoco.MjModel.from_xml_path(str(scene)); d = mujoco.MjData(m)
            for q in ([0]*7, [.2, -.3, .4, .5, -.2, .1, -.4]):
                for width in (0., .04, .1):
                    for i, v in enumerate(q, 1):
                        d.qpos[m.joint('joint%d' % i).qposadr[0]] = v
                    for name, value in [('gripper', width), ('gripper_joint1', width/2), ('gripper_joint2', -width/2)]:
                        d.qpos[m.joint(name).qposadr[0]] = value
                    mujoco.mj_forward(m, d)
                    body = m.body('link7').id; site = m.site('grasp_center').id
                    R = d.xmat[body].reshape(3, 3)
                    np.testing.assert_allclose(d.site_xpos[site], d.xpos[body] + R @ expected_p, atol=1e-9)
                    expected_R = np.column_stack([rotate(expected_q, axis) for axis in np.eye(3)])
                    np.testing.assert_allclose(d.site_xmat[site].reshape(3, 3), R @ expected_R, atol=1e-9)
                    tips = (d.xpos[m.body('gripper_link1').id] + d.xpos[m.body('gripper_link2').id])/2
                    tool_R = d.site_xmat[site].reshape(3, 3)
                    np.testing.assert_allclose(d.site_xpos[site], tips - .005*tool_R[:, 2], atol=1e-7)
                    if width:
                        outward = d.xpos[m.body('gripper_link1').id] - tips
                        np.testing.assert_allclose(outward/np.linalg.norm(outward), tool_R[:, 0], atol=1e-6)

    def test_legacy_migration_preserves_reference_flange_pose(self):
        transform = tcp_in_link7(ET.parse(DEFAULT_MODEL).getroot())
        p, q = transform
        inverse_q = (-q[0], -q[1], -q[2], q[3])
        inverse = (rotate(inverse_q, [-v for v in p]), inverse_q)
        for orientation in ([1., 0., 0., 0.], from_rpy([.3, -.5, 1.2])):
            old = {'frame': 'base_link', 'position_m': [.2, -.3, .4], 'orientation_xyzw': orientation}
            new = migrate_flange_pose(old, transform)
            recovered = compose((new['position_m'], new['orientation_xyzw']), inverse)
            np.testing.assert_allclose(recovered[0], old['position_m'], atol=1e-12)
            np.testing.assert_allclose(recovered[1], orientation, atol=1e-12)
        new = migrate_flange_pose(dict(old, orientation_mode='camera_down_free_yaw'), transform)
        self.assertEqual(new['orientation_mode'], 'fixed')

    def test_rejects_moving_reference_chain(self):
        root = ET.parse(DEFAULT_MODEL).getroot()
        root.find('./joint[@name="gripper_base_joint"]').set('type', 'prismatic')
        with self.assertRaisesRegex(ValueError, 'opening'):
            tcp_in_link7(root)
