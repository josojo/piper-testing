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


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "models/nero/nero_description.urdf"
DEFAULT_OUTPUT = ROOT / "models/nero/nero_scene.xml"
JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "joint7"]


def build(model_path: Path, output_path: Path) -> None:
    if not model_path.is_file():
        raise SystemExit(f"Model not found: {model_path}\nRun scripts/prepare_nero_mujoco.py first")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    # Save the compiled URDF as MJCF so the scene can be extended with normal
    # MJCF elements. Saving beside the source keeps mesh paths valid.
    mujoco.mj_saveLastXML(str(output_path), model)
    tree = ET.parse(output_path)
    root = tree.getroot()
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

    # Detailed imported collision meshes are not tuned for stable dynamics;
    # leave them visual-only for this scene. Collision proxies will be added
    # during the planning/collision-validation phase.
    for geom in worldbody.findall(".//geom"):
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

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
