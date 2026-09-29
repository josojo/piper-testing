#!/usr/bin/env python3
"""Explicit one-time conversion of legacy link7 Cartesian goals to the gripping TCP.

Input must contain legacy flange goals. Never run this twice on the same data.
Free-yaw goals become fixed at their reference orientation: changing the tool
axes cannot preserve their old camera-down constraint.
"""
import argparse
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nero_agent.tool_frame import migrate_flange_pose, tcp_in_link7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--model', type=Path, default=Path(__file__).resolve().parents[1] / 'models/nero/nero_description.urdf')
    args = parser.parse_args()
    config = json.loads(args.input.read_text())
    transform = tcp_in_link7(ET.parse(args.model).getroot())
    for name, pose in config['named_poses'].items():
        if 'position_m' in pose:
            config['named_poses'][name] = migrate_flange_pose(pose, transform)
            print('Converted', name, '(fixed reference orientation)')
    # Exclusive creation protects both source and any previous migration output.
    with args.output.open('x') as stream:
        stream.write(json.dumps(config, indent=2) + '\n')


if __name__ == '__main__':
    main()
