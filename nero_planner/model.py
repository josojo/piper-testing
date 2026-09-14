"""MuJoCo kinematics and conservative box-clearance validation."""

from __future__ import annotations

import hashlib
import itertools
from pathlib import Path

import mujoco
import numpy as np


JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8))
OPEN_GRIPPER = {"gripper": 0.1, "gripper_joint1": 0.05, "gripper_joint2": -0.05}

# Structural pairs share a joint or the base mounting interface. These are
# exempt from clearance, not just penetration, because their housings meet.
STRUCTURAL_BODY_PAIRS = frozenset(frozenset(pair) for pair in (
    ("world", "link1"), ("link1", "link2"), ("link2", "link3"),
    ("link3", "link4"), ("link4", "link5"), ("link5", "link6"),
    ("link6", "link7"), ("link7", "gripper_link1"),
    ("link7", "gripper_link2"),
    # The shoulder's joint1/joint2 housings and the compact joint6/joint7
    # wrist assembly have overlapping enclosing boxes across an intermediate
    # joint body. These named assembly exceptions are deliberate and audited.
    ("world", "link2"), ("link5", "link7"),
))


def box_clearances(centers, rotations, sizes, first, second):
    """Signed conservative distance lower bounds using the 15 OBB SAT axes.

    Positive separation on a unit axis is a lower bound on Euclidean distance.
    A nonpositive maximum means touching/overlapping boxes. This deliberately
    avoids legacy MuJoCo positive-distance inaccuracies for convex colliders.
    """
    axes_a = rotations[first].transpose(0, 2, 1)
    axes_b = rotations[second].transpose(0, 2, 1)
    cross = np.cross(axes_a[:, :, None, :], axes_b[:, None, :, :]).reshape(-1, 9, 3)
    axes = np.concatenate((axes_a, axes_b, cross), axis=1)
    norms = np.linalg.norm(axes, axis=2)
    valid = norms > 1e-10
    axes = axes / np.where(valid, norms, 1)[..., None]
    radius_a = np.sum(np.abs(np.einsum("pai,pji->paj", axes, axes_a)) * sizes[first, None, :], axis=2)
    radius_b = np.sum(np.abs(np.einsum("pai,pji->paj", axes, axes_b)) * sizes[second, None, :], axis=2)
    gap = np.abs(np.einsum("pai,pi->pa", axes, centers[second] - centers[first])) - radius_a - radius_b
    gap[~valid] = -np.inf
    return np.max(gap, axis=1) - 1e-9  # roundoff allowance, metres


class Scene:
    def __init__(self, path: Path):
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)
        m = self.model
        ids = [self.require(mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        if any(m.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE or not m.jnt_limited[i] for i in ids):
            raise ValueError("The seven arm joints must be limited hinges")
        self.qadr = np.array([m.jnt_qposadr[i] for i in ids])
        self.dadr = np.array([m.jnt_dofadr[i] for i in ids])
        self.ranges = m.jnt_range[ids].copy()
        if not np.isfinite(self.ranges).all():
            raise ValueError("Joint ranges must be finite")
        self.gripper = {int(m.jnt_qposadr[self.require(mujoco.mjtObj.mjOBJ_JOINT, name)]): value
                        for name, value in OPEN_GRIPPER.items()}
        if m.nq != 10 or m.nv != 10 or m.nmocap:
            raise ValueError("Planning scene must have seven arm and three gripper joints; obstacles must be fixed")
        self.site = self.require(mujoco.mjtObj.mjOBJ_SITE, "grasp_center")
        for name in ("table", "table_top", "collision_apple"):
            self.require(mujoco.mjtObj.mjOBJ_GEOM, name)
        collision_ids = [i for i in range(m.ngeom)
                         if m.geom(i).name.startswith("collision_") or m.geom(i).name in ("table", "table_top")
                         or m.geom_contype[i] or m.geom_conaffinity[i]]
        if any(m.geom_type[i] != mujoco.mjtGeom.mjGEOM_BOX for i in collision_ids):
            raise ValueError("All planning collision geometry must be boxes")
        covered = {m.body(m.geom_bodyid[i]).name for i in collision_ids if m.geom(i).name.startswith("collision_")}
        if not {"world", *(f"link{i}" for i in range(1, 8)), "gripper_link1", "gripper_link2"} <= covered:
            raise ValueError("Robot collision proxies are missing; rebuild the planning scene")
        self.pairs = []
        for a, b in itertools.combinations(collision_ids, 2):
            ba, bb = (m.body(m.geom_bodyid[i]).name for i in (a, b))
            # No environment/environment tests; only the robot moves. The base
            # is a world-body proxy, so distinguish it by geom name.
            a_robot = m.geom(a).name.startswith(f"collision_{ba}_")
            b_robot = m.geom(b).name.startswith(f"collision_{bb}_")
            if not (a_robot or b_robot):
                continue
            if a_robot and b_robot and (ba == bb or frozenset((ba, bb)) in STRUCTURAL_BODY_PAIRS):
                continue
            if ((a_robot and ba == "world" and m.geom(b).name in ("table", "table_top"))
                    or (b_robot and bb == "world" and m.geom(a).name in ("table", "table_top"))):
                continue
            self.pairs.append((a, b))
        self.first, self.second = np.array(self.pairs, dtype=int).T
        # Any robot point is at most R from any ancestor hinge. This loose
        # global bound includes all fixed translations, anchor offsets, box
        # radii and possible prismatic offsets. Rotation preserves lengths.
        self.motion_radius = float(
            np.linalg.norm(m.body_pos, axis=1).sum()
            + 2 * np.linalg.norm(m.jnt_pos, axis=1).sum()
            + np.max(np.linalg.norm(m.geom_pos, axis=1) + np.linalg.norm(m.geom_size, axis=1))
            + sum(abs(value) for value in OPEN_GRIPPER.values())
        )
        self.set_arm(np.zeros(7))

    def require(self, kind, name):
        index = mujoco.mj_name2id(self.model, kind, name)
        if index < 0:
            raise ValueError(f"Required model element is missing: {name}")
        return index

    def set_arm(self, q):
        self.data.qpos[self.qadr] = q
        for address, value in self.gripper.items():
            self.data.qpos[address] = value
        self.data.qvel[:] = 0
        mujoco.mj_forward(self.model, self.data)

    def distances(self, q):
        self.set_arm(q)
        return box_clearances(self.data.geom_xpos, self.data.geom_xmat.reshape(-1, 3, 3),
                              self.model.geom_size, self.first, self.second)

    def pose(self, q):
        self.set_arm(q)
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.site])
        return self.data.site_xpos[self.site].copy(), quat

    def fingerprint(self):
        """Hash model state that can affect geometry, kinematics or policy."""
        digest = hashlib.sha256()
        for name in ("body_pos", "body_quat", "body_parentid", "jnt_pos", "jnt_axis",
                     "jnt_range", "jnt_type", "jnt_limited", "jnt_qposadr", "jnt_dofadr", "jnt_bodyid",
                     "geom_pos", "geom_quat", "geom_size", "geom_type", "geom_bodyid",
                     "geom_contype", "geom_conaffinity",
                     "site_pos", "site_quat", "site_bodyid", "qpos0"):
            digest.update(np.asarray(getattr(self.model, name)).tobytes())
        digest.update(bytes(self.model.names))
        digest.update(np.asarray(self.pairs).tobytes())
        return digest.hexdigest()
