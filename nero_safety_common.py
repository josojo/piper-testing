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
    """Return the native CandleLight configuration for this setup.

    ``candle`` is available as an optional backend for newer CandleLight
    devices, but this adapter exposes the legacy 0x1d50:0x606f protocol.
    """
    if platform.system() != "Darwin":
        raise RuntimeError("This local setup expects macOS (Darwin).")
    interface = os.environ.get("NERO_CAN_INTERFACE", "gs_usb")
    return {
        "interface": interface,
        # python-can treats channel=0 as the first discovered gs_usb device.
        "channel": 0,
        "bitrate": 1_000_000,
        "auto_connect": False,
    }


def connect_nero():
    """Connect to Nero through CandleLight/gs_usb on macOS."""
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
