"""Shared CAN connection setup for Nero safety commands."""

import platform
import os
import time

from pyAgxArm import (
    AgxArmFactory,
    ArmModel,
    NeroFW,
    create_agx_arm_config,
)


def nero_can_config():
    """Return the pyAgxArm CAN configuration for the current host.

    pyAgxArm's documented Ubuntu setup uses a SocketCAN interface that must
    already be activated by the operating system.  macOS continues to use
    the serial-line CAN backend.
    """
    system = platform.system()
    if system == "Linux":
        defaults = {"interface": "socketcan", "channel": "can0"}
    elif system == "Darwin":
        defaults = {"interface": "slcan", "channel": "/dev/ttyACM0"}
    else:
        raise RuntimeError(
            "Unsupported host OS. pyAgxArm documents Linux SocketCAN and "
            "macOS slcan for this setup."
        )

    interface = os.environ.get("NERO_CAN_INTERFACE", defaults["interface"])
    channel = os.environ.get("NERO_CAN_CHANNEL", defaults["channel"])
    return {
        "interface": interface,
        "channel": channel,
        "bitrate": 1_000_000,
        "auto_connect": False,
    }


def connect_nero():
    """Connect to Nero using the host platform's documented CAN backend."""
    can_config = nero_can_config()
    config = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=NeroFW.V121,
        **can_config,
    )
    robot = AgxArmFactory.create_arm(config)
    robot.connect()
    return robot


def disconnect_nero(robot, settle_time=0.25):
    """Stop the reader before closing the serial CAN adapter."""
    if robot is not None:
        time.sleep(settle_time)
        robot.disconnect(join_timeout=5.0)
