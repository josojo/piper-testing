"""ROS transport for NERO: SDK feedback, bounded controller streaming, no planning.

No auto-enable, reset, homing, gripper motion, or idle motion writes. The
ros2_control JointTrajectoryController supplies the timed positions.
"""
import argparse
from copy import deepcopy
import json
import math
import time

from .core import AgentError, JOINTS
from .stream_guard import StreamGuard


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--namespace', default='/nero')
    args = parser.parse_args()
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty, String
    from std_srvs.srv import SetBool, Trigger
    from nero_experiment.hardware import connect

    class Driver(Node):
        def __init__(self, hardware):
            super().__init__('nero_hardware_bridge', namespace=args.namespace)
            self.hardware = hardware
            self.effector = hardware.robot.init_effector(hardware.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
            self.guard = None
            self.last_state = None
            self.last_width = None
            self.received = 0.0
            self.fault = None
            self.feedback_error = None
            self.fault_pub = self.create_publisher(String, 'project/fault', 10)
            self.publisher = self.create_publisher(JointState, 'feedback/joint_states', 10)
            self.create_subscription(JointState, 'control/joint_commands', self.command, 1)
            self.create_subscription(Empty, 'project/heartbeat', self.heartbeat, 1)
            self.create_service(Trigger, 'project/info', self.info)
            self.create_service(Trigger, 'project/stop', self.stop_service)
            self.create_service(SetBool, 'project/control_enable', self.gate)
            self.create_timer(0.01, self.tick)

        def gripper(self):
            packet = deepcopy(self.effector.get_gripper_status())
            if packet is None or not 0 <= time.time() - packet.timestamp <= 0.25:
                raise AgentError('Missing/stale gripper feedback; collision geometry requires measured opening')
            width = packet.msg.value
            if not math.isfinite(width) or not 0 <= width <= 0.1:
                raise AgentError('Invalid gripper opening')
            return float(width), packet.timestamp

        def tick(self):
            if self.fault:
                self.fault_pub.publish(String(data=self.fault))
            try:
                state = self.hardware.read(require_enabled=self.guard is not None)
                width, grip_stamp = self.gripper()
                now = time.monotonic()
                if self.guard:
                    self.guard.check_feedback(state.joints_rad, state.velocities_rad_s, width, now)
                self.last_state, self.last_width, self.received = state, width, now
                msg = JointState()
                # Use the oldest contributing packet, not the publication time.
                stamp = min(*state.feedback_timestamps[:4], grip_stamp)
                msg.header.stamp.sec = int(stamp)
                msg.header.stamp.nanosec = int((stamp - int(stamp)) * 1e9)
                msg.name = list(JOINTS) + ['gripper']
                msg.position = list(state.joints_rad) + [width]
                msg.velocity = list(state.velocities_rad_s) + [0.0]
                self.publisher.publish(msg)
                self.feedback_error = None
            except Exception as error:
                detail = str(error) or type(error).__name__
                if detail != self.feedback_error:
                    self.get_logger().warning('Feedback unavailable: ' + detail)
                self.feedback_error = detail
                if self.guard:
                    self.fault = str(error)
                    try:
                        self.stop()
                    except Exception as stop_error:
                        self.fault += '; STOP DELIVERY FAILED: ' + str(stop_error)
                    self.get_logger().error(self.fault)
                    # Stop first; diagnostics must not delay stop delivery.
                    if hasattr(error, 'history'):
                        self.get_logger().error('velocity_trip_history=' + json.dumps(error.history))

        def info(self, request, response):
            response.success = (self.last_state is not None and time.monotonic() - self.received < 0.25
                                and not self.fault and not self.feedback_error)
            response.message = (json.dumps({'mode': 'hardware', 'effector': 'agx_gripper',
                'armed': self.guard is not None, 'streaming_mode': 'move_js'}) if response.success else
                self.fault or self.feedback_error or 'Waiting for complete arm and gripper feedback')
            return response

        def heartbeat(self, msg):
            if self.guard:
                self.guard.heartbeat_at = time.monotonic()

        def stop(self):
            self.guard = None  # Reject future commands even if CAN stop fails.
            self.hardware.stop()

        def stop_service(self, request, response):
            try:
                self.stop()
                response.success, response.message = True, 'Electronic stop sent; gate closed'
            except Exception as error:
                response.success, response.message = False, str(error)
            return response

        def gate(self, request, response):
            if request.data and (self.guard or self.fault):
                response.success = False
                response.message = self.fault or 'Execution gate already open'
                return response
            try:
                if request.data:
                    self.hardware.stationary(require_enabled=True)
                    self.hardware.configure_acceleration()
                    state = self.hardware.stationary(require_enabled=True)
                    width, _ = self.gripper()
                    self.last_state, self.last_width, self.received = state, width, time.monotonic()
                    self.guard = StreamGuard(state.joints_rad, width, self.received)
                    # Setup uses blocking SDK reads. Refresh publication before
                    # acknowledging the gate so clients do not inherit old state.
                    self.tick()
                    if self.fault:
                        raise AgentError(self.fault)
                else:
                    self.guard = None  # Successful arrival: retain position, do not disable.
                response.success, response.message = True, 'Gate open' if request.data else 'Gate closed'
            except Exception as error:
                self.guard = None
                response.success, response.message = False, str(error)
            return response

        def command(self, msg):
            if not self.guard:
                return
            try:
                if self.last_state is None or time.monotonic() - self.received > 0.05:
                    raise AgentError('No fresh measured state for command')
                if len(msg.name) != len(msg.position) or len(set(msg.name)) != len(msg.name):
                    raise AgentError('Malformed controller joint command: %d names, %d positions' %
                                     (len(msg.name), len(msg.position)))
                by_name = dict(zip(msg.name, msg.position))
                q = [by_name[n] for n in JOINTS]
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                q = self.guard.command(q, self.last_state.joints_rad, stamp, time.time(), time.monotonic())
                # Streaming mode follows controller interpolation; no move_j replanning.
                # The SDK speed percentage does not define this stream's timing.
                self.hardware.robot.move_js(list(q))
            except Exception as error:
                self.fault = str(error)
                try:
                    self.stop()
                except Exception as stop_error:
                    self.fault += '; STOP DELIVERY FAILED: ' + str(stop_error)
                self.get_logger().error(self.fault)

    rclpy.init()
    node = None
    try:
        with connect() as hardware:
            node = Driver(hardware)
            try:
                rclpy.spin(node)
            finally:
                if node.guard:
                    node.stop()
                node.destroy_node()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
