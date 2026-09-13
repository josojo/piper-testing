"""Shared CAN connection setup for Nero safety commands."""

import time

from pyAgxArm import (
    AgxArmFactory,
    ArmModel,
    NeroFW,
    create_agx_arm_config,
)


def connect_nero():
    """Connect to the Nero using the local gs_usb adapter."""
    config = create_agx_arm_config(
        robot=ArmModel.NERO,
        firmeware_version=NeroFW.V121,
        interface="gs_usb",
        channel=0,
        bitrate=1_000_000,
    )
    robot = AgxArmFactory.create_arm(config)
    robot.connect()
    return robot


def disconnect_nero(robot, settle_time=0.25):
    """Stop the driver cleanly before another process reopens the USB adapter.

    pyAgxArm's reader can be inside a one-second USB receive when its default
    one-second join timeout expires.  Closing the gs_usb handle at that point
    can leave the adapter in a state where the next process opens it but its
    first transmit times out.
    """
    if robot is not None:
        time.sleep(settle_time)
        robot.disconnect(join_timeout=2.5)
