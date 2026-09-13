#!/usr/bin/env python3
"""Send Nero's controlled electronic emergency-stop command."""

from nero_safety_common import connect_nero, disconnect_nero


def main():
    robot = None
    try:
        print("Connecting to Nero...")
        robot = connect_nero()
        print("Connected")
        print("Sending controlled electronic emergency stop...")
        robot.electronic_emergency_stop()
        print("Emergency-stop command sent")
    finally:
        if robot is not None:
            disconnect_nero(robot)
            print("Disconnected")


if __name__ == "__main__":
    main()
