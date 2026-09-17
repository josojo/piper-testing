"""ROS transport for NERO: SDK feedback and ROS2 trajectory-controller bridge.

No auto-enable, reset, homing, gripper motion, or idle motion writes. The
ros2_control JointTrajectoryController supplies the timed positions.
"""
import argparse
from copy import deepcopy
import json
import math
import time

from .core import AgentError, JOINTS, MAX_EXCURSION_RAD, LARGE_EXCURSION_RAD, SEGMENT_EXCURSION_RAD
from .stream_guard import StreamGuard
from .controlled_abort import ControlledAbort


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--commission-abort', action='store_true')
    parser.add_argument('--diagnose-feedback', action='store_true')
    parser.add_argument('--diagnostic-duration', type=float, default=30.)
    parser.add_argument('--abort-qualification-report', type=str)
    parser.add_argument('--motor-velocity-limit', type=float, default=0.10,
                        help='Temporary motor-feedback trip limit in rad/s (0.10-0.15)')
    parser.add_argument('--namespace', default='/nero')
    profile = parser.add_mutually_exclusive_group()
    profile.add_argument('--large-motion', action='store_true')
    profile.add_argument('--segmented-motion', action='store_true')
    args = parser.parse_args()
    if not math.isfinite(args.diagnostic_duration) or not 1 <= args.diagnostic_duration <= 120:
        parser.error('diagnostic duration must be between 1 and 120 seconds')
    if not math.isfinite(args.motor_velocity_limit) or not 0.10 <= args.motor_velocity_limit <= 0.15:
        parser.error('motor velocity limit must be between 0.10 and 0.15 rad/s')
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty, String
    from std_srvs.srv import SetBool, Trigger
    from action_msgs.srv import CancelGoal
    from nero_experiment.hardware import MAX_JOINT_SNAPSHOT_AGE_S, connect

    class Driver(Node):
        def __init__(self, hardware):
            super().__init__('nero_hardware_bridge', namespace=args.namespace)
            self.hardware = hardware
            self.effector = hardware.robot.init_effector(hardware.robot.OPTIONS.EFFECTOR.AGX_GRIPPER)
            self.diagnostic_trial = None
            self.report_cache = None
            self.recorder = None
            self.diagnostic_end = None
            self.guard = None
            self.trial = None
            self.last_state = None
            self.last_width = None
            self.received = 0.0
            self.fault = None
            self.feedback_error = None
            self.velocity_trip_history = None
            self.cancel_client = self.create_client(CancelGoal, 'arm_controller/follow_joint_trajectory/_action/cancel_goal')
            self.abort = ControlledAbort(hardware, self.block_stream, self.cancel_controller)
            self.fault_pub = self.create_publisher(String, 'project/fault', 10)
            self.publisher = self.create_publisher(JointState, 'feedback/joint_states', 10)
            self.create_subscription(JointState, 'control/joint_commands', self.command, 1)
            self.create_subscription(Empty, 'project/heartbeat', self.heartbeat, 1)
            self.create_service(Trigger, 'project/commission_abort', self.commission_gate)
            if args.diagnose_feedback:
                self.create_service(Trigger, 'project/diagnostic_start', self.diagnostic_start)
                self.create_service(Trigger, 'project/diagnostic_result', self.diagnostic_result)
            self.create_service(Trigger, 'project/info', self.info)
            self.create_service(Trigger, 'project/stop', self.stop_service)
            self.create_service(Trigger, 'project/abort_status', self.abort_status)
            self.create_service(Trigger, 'project/abort_report', self.abort_report)
            self.create_service(Trigger, 'project/emergency_stop', self.emergency_stop)
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

        def diagnostic_start(self, request, response):
            from .feedback_diagnostic import Recorder
            response.success = self.recorder is None
            response.message = 'Diagnostic already started'
            if response.success:
                from .commissioning import MovingAbortTrial
                self.diagnostic_trial = MovingAbortTrial([0.] * 7, time.monotonic())
                self.diagnostic_trial.outcome = 'diagnostic_only'
                self.recorder = Recorder(self.hardware)
                self.diagnostic_end = time.monotonic() + args.diagnostic_duration
                response.message = 'Recording'
            return response

        def diagnostic_result(self, request, response):
            response.success = self.recorder is not None and time.monotonic() >= self.diagnostic_end
            response.message = 'Diagnostic not finished'
            if response.success:
                result = self.recorder.result()
                result['scope'] = 'Full idle MoveIt/ros2_control stack; measured inside bridge timer. Hardware writes blocked. SDK cache polling, not raw CAN capture.'
                response.message = json.dumps(result)
            return response

        def tick(self):
            recording = self.recorder is not None and time.monotonic() < self.diagnostic_end
            if recording:
                self.recorder.begin()
            try:
                self.feedback_tick(recording)
            finally:
                if recording:
                    self.recorder.finish()

        def feedback_tick(self, recording=False):
            self.abort.tick()
            if self.fault:
                self.fault_pub.publish(String(data=self.fault))
            try:
                state = (self.recorder if recording else self.hardware).read(require_enabled=self.guard is not None)
                width, grip_stamp = self.gripper()
                now = time.monotonic()
                if recording:
                    self.diagnostic_trial.observe(state, now)
                trial_reason = self.trial.observe(state, now) if self.trial else None
                if self.guard:
                    self.guard.check_feedback(
                        state.joints_rad, state.velocities_rad_s, width, now,
                        velocity_timestamps=state.motor_velocity_timestamps,
                        wall_now=time.time(),
                        position_timestamps=state.joint_position_timestamps)
                if trial_reason:
                    if self.trial.outcome == 'failed':
                        self.fault = trial_reason
                    self.abort.start(trial_reason)
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
                if recording:
                    self.recorder.current['bridge_error'] = detail
                if detail != self.feedback_error:
                    self.get_logger().warning('Feedback unavailable: ' + detail)
                self.feedback_error = detail
                if self.trial is not None:
                    self.fault = detail
                if self.guard:
                    self.fault = str(error)
                    if hasattr(error, 'history'):
                        self.velocity_trip_history = error.history
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
                'armed': self.guard is not None, 'streaming_mode': 'ros2_joint_trajectory_controller'}) if response.success else
                self.fault or self.feedback_error or 'Waiting for complete arm and gripper feedback')
            return response

        def heartbeat(self, msg):
            if self.guard:
                self.guard.heartbeat_at = time.monotonic()

        def stop(self):
            return self.abort.start(self.fault or 'Operator/application controlled abort')

        def block_stream(self):
            self.guard = None

        def cancel_controller(self):
            if not self.cancel_client.service_is_ready():
                raise AgentError('Controller cancellation service unavailable')
            # Zero UUID/stamp cancels all goals on this dedicated arm controller.
            return self.cancel_client.call_async(CancelGoal.Request())

        def abort_result(self, full=False):
            result = self.abort.result()
            if full and self.velocity_trip_history is not None:
                result['velocity_trip_history'] = self.velocity_trip_history
            trial = self.trial or self.diagnostic_trial
            if trial:
                result['commissioning'] = (trial.result(result, self.fault) if full else trial.live_status())
            return result

        def abort_report(self, request, response):
            diagnostic_done = (args.diagnose_feedback and self.diagnostic_end is not None
                               and time.monotonic() >= self.diagnostic_end)
            response.success = self.abort.phase in ('holding', 'rechecking', 'failed') or diagnostic_done
            response.message = 'Full trace unavailable until observation ends'
            if response.success:
                # Reuse the serialized snapshot for repeated report requests.
                cache_key = (self.abort.phase, self.abort.samples, self.abort.recheck_count)
                if self.report_cache is None or self.report_cache[0] != cache_key:
                    self.report_cache = (cache_key, json.dumps(self.abort_result(full=True)))
                response.message = self.report_cache[1]
            return response

        def commission_gate(self, request, response):
            if args.diagnose_feedback:
                response.success, response.message = False, 'Hardware writes blocked in feedback diagnostic mode'
                return response
            try:
                if not args.commission_abort or self.trial is not None or self.guard or self.fault or self.abort.phase != 'idle':
                    raise AgentError('Single-use commissioning gate unavailable')
                from .commissioning import MovingAbortTrial
                state = self.hardware.stationary(require_enabled=True)
                if max(map(abs, state.velocities_rad_s)) > .003:
                    raise AgentError('Commissioning requires stationary enabled joints')
                width, _ = self.gripper()
                now = time.monotonic()
                self.trial = MovingAbortTrial(state.joints_rad, now)
                self.abort.STOPPED_SPEED = .003
                self.abort.HOLD_DWELL = 2.0
                self.abort.POSITION_TOLERANCE = .002
                self.abort.MAX_EXCURSION = .01
                self.last_state, self.last_width, self.received = state, width, now
                self.guard = StreamGuard(state.joints_rad, width, now, args.motor_velocity_limit)
                self.tick()
                if self.fault:
                    raise AgentError(self.fault)
                response.success, response.message = True, 'Restricted commissioning gate open'
            except Exception as error:
                response.success, response.message = False, str(error)
            return response

        def abort_status(self, request, response):
            response.success = self.abort.phase != 'failed'
            response.message = json.dumps(self.abort_result())
            return response

        def emergency_stop(self, request, response):
            if args.diagnose_feedback:
                response.success, response.message = False, 'Hardware writes blocked in feedback diagnostic mode'
                return response
            self.block_stream()
            self.abort.fail('Explicit emergency stop requested; powered hold abandoned')
            try:
                self.hardware.stop()
                response.success = True
                response.message = 'Damped electronic stop sent; this can permit descent'
            except Exception as error:
                response.success, response.message = False, str(error)
            return response

        def stop_service(self, request, response):
            if args.diagnose_feedback:
                response.success, response.message = False, 'Hardware writes blocked in feedback diagnostic mode'
                return response
            try:
                self.stop()
                response.success = self.abort.phase != 'failed'
                response.message = json.dumps(self.abort_result())
            except Exception as error:
                response.success, response.message = False, str(error)
            return response

        def gate(self, request, response):
            if args.diagnose_feedback:
                response.success, response.message = False, 'Hardware writes blocked in feedback diagnostic mode'
                return response
            if request.data and self.abort.phase != 'idle':
                response.success, response.message = False, 'Controlled abort latched; restart requires operator review'
                return response
            if request.data:
                from .abort_policy import require_verified_controlled_abort
                try:
                    require_verified_controlled_abort(args.abort_qualification_report)
                except AgentError as error:
                    response.success, response.message = False, str(error)
                    return response
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
                    self.guard = StreamGuard(state.joints_rad, width, self.received,
                                             args.motor_velocity_limit,
                                             SEGMENT_EXCURSION_RAD if args.segmented_motion else
                                             LARGE_EXCURSION_RAD if args.large_motion else MAX_EXCURSION_RAD)
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
            if args.diagnose_feedback or not self.guard or self.abort.phase != 'idle':
                return
            try:
                if (self.last_state is None
                        or time.monotonic() - self.received > MAX_JOINT_SNAPSHOT_AGE_S):
                    raise AgentError('No fresh measured state for command')
                if len(msg.name) != len(msg.position) or len(set(msg.name)) != len(msg.name):
                    raise AgentError('Malformed controller joint command: %d names, %d positions' %
                                     (len(msg.name), len(msg.position)))
                by_name = dict(zip(msg.name, msg.position))
                q = [by_name[n] for n in JOINTS]
                stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                q = self.guard.command(q, self.last_state.joints_rad, stamp, time.time(), time.monotonic())
                if self.trial:
                    self.trial.command(q, stamp, time.monotonic())
                # The ROS2 joint_trajectory_controller supplies timed
                # interpolation. Forward its position targets through the
                # vendor's ordinary MOVE_J position interface; MOVE_JS is an
                # instantaneous, unsmoothed mode and must not be used here.
                self.hardware.robot.move_j(list(q))
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
                # On shutdown callbacks may no longer spin. Cancellation times
                # out boundedly; fresh SDK feedback still drives the hold check.
                deadline = time.monotonic() + 12.0
                while node.abort.phase not in ('idle', 'holding', 'failed') and time.monotonic() < deadline:
                    node.abort.tick()
                    time.sleep(0.01)
                if node.abort.phase != 'idle':
                    node.get_logger().warning('Final controlled abort status: ' + json.dumps(node.abort.result()))
                node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
