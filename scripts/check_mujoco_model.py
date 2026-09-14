#!/usr/bin/env python3
"""Load the prepared NERO URDF with MuJoCo and print basic model facts."""

from pathlib import Path

import mujoco


MODEL = Path(__file__).resolve().parents[1] / "models/nero/nero_description.urdf"


def main() -> None:
    if not MODEL.is_file():
        raise SystemExit(f"Model not found: {MODEL}\nRun: python scripts/prepare_nero_mujoco.py")
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    joints = [model.joint(i).name for i in range(model.njnt)]
    print(f"Loaded {MODEL}")
    print(f"joints={model.njnt} dofs={model.nv} bodies={model.nbody} geoms={model.ngeom}")
    print(f"joint_names={joints}")


if __name__ == "__main__":
    main()

