#!/usr/bin/env python3
"""Fetch the official NERO description and prepare a MuJoCo-loadable URDF.

This does not alter the hardware-control scripts.  The downloaded source is
kept outside the generated model so it can be updated independently.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


REPOSITORY = "https://github.com/agilexrobotics/agx_arm_urdf.git"
DEFAULT_SOURCE = Path("models/official/agx_arm_urdf")
DEFAULT_OUTPUT = Path("models/nero/nero_description.urdf")


def fetch_repository(destination: Path) -> None:
    if (destination / ".git").is_dir():
        print(f"Using existing source: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {REPOSITORY} -> {destination}")
    subprocess.run(["git", "clone", "--depth", "1", REPOSITORY, str(destination)], check=True)


def expand_xacro(source: Path, output: Path) -> None:
    xacro = shutil.which("xacro")
    if not xacro:
        raise RuntimeError("xacro was not found; install ROS xacro or omit --xacro")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        subprocess.run([xacro, str(source)], check=True, stdout=stream)


def make_absolute_mesh_paths(urdf: Path, model_root: Path, output: Path) -> None:
    """Resolve package:// and relative mesh URIs for standalone MuJoCo use."""
    text = urdf.read_text(encoding="utf-8")
    package_prefix = "package://agx_arm_description/agx_arm_urdf/nero/"
    absolute_model_root = model_root.resolve()
    text = text.replace(package_prefix, str(absolute_model_root) + "/")
    text = text.replace("package://agx_arm_urdf/nero/", str(absolute_model_root) + "/")

    # The official URDF normally uses paths relative to its urdf/ directory.
    # Keep other XML untouched and resolve only mesh filenames.
    import re

    output.parent.mkdir(parents=True, exist_ok=True)

    def resolve(match: re.Match[str]) -> str:
        uri = match.group(1)
        if "://" in uri:
            return match.group(0)
        candidate = Path(uri) if Path(uri).is_absolute() else (urdf.parent / uri).resolve()
        if candidate.is_file():
            # MuJoCo's URDF importer uses the generated URDF directory when
            # resolving mesh basenames. Stage a local copy for deterministic
            # loading, while retaining the official checkout as the source.
            staged = output.parent / candidate.name
            if staged.resolve() != candidate.resolve():
                shutil.copy2(candidate, staged)
            return f'filename="{candidate.name}"'
        return match.group(0)

    text = re.sub(r'filename="([^"]+)"', resolve, text)
    output.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="official agx_arm_urdf checkout")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                        help="generated standalone URDF")
    parser.add_argument("--xacro", action="store_true",
                        help="expand nero_with_gripper_description.xacro via xacro")
    parser.add_argument("--no-clone", action="store_true",
                        help="require --source to already exist")
    args = parser.parse_args()

    source = args.source
    if not args.no_clone:
        fetch_repository(source)
    urdf_dir = source / "nero" / "urdf"
    input_path = urdf_dir / ("nero_with_gripper_description.xacro" if args.xacro
                             else "nero_description.urdf")
    if not input_path.is_file():
        print(f"Missing official model file: {input_path}", file=sys.stderr)
        return 2
    if args.xacro:
        expanded = args.output.with_suffix(".expanded.urdf")
        expand_xacro(input_path, expanded)
        input_path = expanded
    make_absolute_mesh_paths(input_path, source / "nero", args.output)
    print(f"Prepared: {args.output.resolve()}")
    print("Next: python scripts/check_mujoco_model.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
