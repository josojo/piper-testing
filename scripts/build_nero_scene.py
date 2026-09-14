#!/usr/bin/env python3
"""Build a small MJCF scene with the prepared NERO model and table.

The URDF is compiled by MuJoCo first, then the table and position actuators
are added to the resulting MJCF. The generated XML remains ignored.
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/nero/nero_description.urdf"
DEFAULT_OUTPUT = ROOT / "models/nero/nero_scene.xml"
JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]
GRIPPER_JOINT = "gripper"
PROXY_PADDING_M = 0.003


def numbers(values) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def add_planning_geometry(root: ET.Element, model: mujoco.MjModel) -> None:
    """Enclose every imported mesh with a padded box in its compiled frame."""
    world = root.find("worldbody")
    bodies = {body.get("name"): body for body in world.findall(".//body")}
    bodies["world"] = world
    for geom_id in range(model.ngeom):
        if model.geom_type[geom_id] != mujoco.mjtGeom.mjGEOM_MESH:
            raise ValueError("Expected mesh geometry in the vendor model")
        mesh_id = model.geom_dataid[geom_id]
        start = model.mesh_vertadr[mesh_id]
        vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh_id]]
        lower, upper = vertices.min(axis=0), vertices.max(axis=0)
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, model.geom_quat[geom_id])
        center = model.geom_pos[geom_id] + rotation.reshape(3, 3) @ ((lower + upper) / 2)
        body_name = model.body(model.geom_bodyid[geom_id]).name
        ET.SubElement(bodies[body_name], "geom", {
            "name": f"collision_{body_name}_{geom_id}", "type": "box",
            "pos": numbers(center), "quat": numbers(model.geom_quat[geom_id]),
            "size": numbers((upper - lower) / 2 + PROXY_PADDING_M),
            "rgba": "0.2 0.7 0.9 0.25", "group": "3", "mass": "0",
            "contype": "1", "conaffinity": "1",
        })

    # The two finger origins lie on the grasp-center plane. Use their midpoint
    # at zero opening, with the gripper-base orientation (URDF +Z approach).
    finger = bodies["gripper_link1"]
    ET.SubElement(bodies["link7"], "site", {
        "name": "grasp_center", "pos": finger.get("pos", "0 0 0"),
        "quat": bodies["gripper_link"].get("quat", "1 0 0 0"), "size": "0.006",
        "rgba": "0 1 0 1", "group": "0",
    })
    equality = ET.SubElement(root, "equality")
    for name, multiplier in (("gripper_joint1", 0.5), ("gripper_joint2", -0.5)):
        ET.SubElement(equality, "joint", {
            "joint1": name, "joint2": "gripper", "polycoef": f"0 {multiplier} 0 0 0",
        })


def build(model_path: Path, output_path: Path) -> None:
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}\nRun scripts/prepare_nero_mujoco.py first")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    # Save the compiled URDF as MJCF so the scene can be extended with normal
    # MJCF elements. Saving beside the source keeps mesh paths valid.
    mujoco.mj_saveLastXML(str(output_path), model)
    tree = ET.parse(output_path)
    root = tree.getroot()
    # Keep --output usable outside models/nero (including isolated tests).
    for mesh in root.findall("./asset/mesh"):
        mesh.set("file", str((model_path.resolve().parent / mesh.get("file")).resolve()))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Compiled model has no worldbody")

    # The vendor URDF is a kinematic/geometry description and has no tuned
    # dynamic parameters. Use conservative inspection defaults until a proper
    # NERO dynamics calibration is available.
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(1, option)
    option.set("gravity", "0 0 0")
    option.set("timestep", "0.002")
    for joint in worldbody.findall(".//joint"):
        if joint.get("name") in JOINTS:
            joint.set("damping", "5")
            joint.set("armature", "0.05")

    # Detailed meshes remain visual-only; padded boxes enclose their vertices.
    for geom in worldbody.findall(".//geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")
    add_planning_geometry(root, model)

    ET.SubElement(worldbody, "geom", {
        "name": "table",
        "type": "box",
        "pos": "0 0 -0.06",
        "size": "0.60 0.60 0.05",
        "rgba": "0.35 0.22 0.12 1",
        "contype": "1",
        "conaffinity": "1",
    })
    ET.SubElement(worldbody, "geom", {
        "name": "table_top",
        "type": "box",
        "pos": "0 0 -0.005",
        "size": "0.58 0.58 0.005",
        "rgba": "0.55 0.35 0.18 1",
    })
    apple = ET.SubElement(worldbody, "body", {
        "name": "apple",
        "pos": "0.30 0.00 0.04",
    })
    ET.SubElement(apple, "geom", {
        "name": "apple_fruit",
        "type": "sphere",
        "size": "0.04",
        "rgba": "0.82 0.03 0.02 1",
        "mass": "0.15",
        "contype": "0", "conaffinity": "0",
    })
    ET.SubElement(apple, "geom", {
        "name": "apple_stem",
        "type": "cylinder",
        "pos": "0 0 0.045",
        "size": "0.004 0.012",
        "rgba": "0.20 0.07 0.01 1",
        "mass": "0.005",
        "contype": "0", "conaffinity": "0",
    })
    ET.SubElement(apple, "geom", {
        "name": "collision_apple", "type": "box", "pos": "0 0 0.0085",
        "size": "0.043 0.043 0.0515", "group": "3",
        "rgba": "1 0.5 0 0.25", "mass": "0",
    })

    actuators = ET.SubElement(root, "actuator")
    for joint_name in JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise RuntimeError(f"Expected joint is missing: {joint_name}")
        lower, upper = model.jnt_range[joint_id]
        ET.SubElement(actuators, "position", {
            "name": f"{joint_name}_position",
            "joint": joint_name,
            "kp": "10",
            "kv": "2",
            "ctrlrange": f"{lower:.8g} {upper:.8g}",
            "forcerange": "-20 20",
        })
    gripper_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, GRIPPER_JOINT)
    if gripper_id >= 0:
        ET.SubElement(actuators, "position", {
            "name": "gripper_position",
            "joint": GRIPPER_JOINT,
            "kp": "5",
            "kv": "1",
            "ctrlrange": "0 0.1",
            "forcerange": "-5 5",
        })

    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "headlight", {"ambient": "0.5 0.5 0.5"})
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    # Verify the final scene, including the table and actuators.
    checked = mujoco.MjModel.from_xml_path(str(output_path))
    print(f"Built: {output_path}")
    print(f"joints={checked.njnt} actuators={checked.nu} geoms={checked.ngeom}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build(args.model, args.output)


if __name__ == "__main__":
    main()
