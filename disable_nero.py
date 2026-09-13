#!/usr/bin/env python3
"""Disable all Nero motor joints."""

from nero_safety_common import connect_nero, disconnect_nero


def main():
    robot = None
    try:
        print("Connecting to Nero...")
        robot = connect_nero()
        print("Connected")

        acknowledged = robot.disable()
        states = robot.get_joints_enable_status_list()
        print("Disable command acknowledged:", acknowledged)
        print("Joint enabled states:", states)

        if any(states):
            raise RuntimeError("Some joints still report enabled")
        print("All Nero joints are disabled")
    finally:
        if robot is not None:
            disconnect_nero(robot)
            print("Disconnected")


if __name__ == "__main__":
    main()
