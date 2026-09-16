#!/usr/bin/env python3
"""Disable all Nero motor joints."""

from nero_safety_common import connect_nero, disconnect_nero
from copy import deepcopy
import time


def wait_until_disabled(robot, sent_at, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        packets = [deepcopy(robot.get_driver_states(i)) for i in range(1, 8)]
        now = time.time()
        if all(p is not None and p.timestamp > sent_at and
               0 <= now - p.timestamp <= 0.25 and
               not p.msg.foc_status.driver_enable_status for p in packets):
            return
        time.sleep(0.05)
    raise TimeoutError('Could not confirm all seven joints disabled from fresh feedback')


def main():
    if input('Support the arm: disabling releases holding torque and it may fall. '
             'Stop other controllers first. Type DISABLE to continue: ').strip() != 'DISABLE':
        print('Cancelled; no command sent.')
        return
    robot = None
    try:
        print("Connecting to Nero...")
        robot = connect_nero()
        print("Connected")

        sent_at = time.time()
        robot.disable()
        wait_until_disabled(robot, sent_at)
        print("Fresh feedback confirms all seven Nero joints are disabled")
    finally:
        if robot is not None:
            disconnect_nero(robot)
            print("Disconnected")


if __name__ == "__main__":
    main()
