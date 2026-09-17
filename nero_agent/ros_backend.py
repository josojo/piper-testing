"""MoveIt services/actions with independent measured-state completion checks.

ROS imports stay here so the offline contract test works with ordinary Python.
"""
import hashlib
import json
import math
import time
import uuid

from .core import AgentError, JOINTS, distance, vector, excursion_detail


# Target 0.02 rad/s from the model's 0.08 rad/s cap while characterizing tracking.
# The independent hardware trip threshold is configured separately.
PLANNING_VELOCITY_SCALING = 0.25
# Target 0.03 rad/s² from the model's 0.15 rad/s² acceleration cap.
PLANNING_ACCELERATION_SCALING = 0.2


def configure_goal_orientation(constraint, target):
    """Euler XYZ error relative to reference: constrain tilt, optionally free spin.

    Unlike rotation-vector components, Euler tilt tolerances do not tighten
    with a large allowed spin. This is an endpoint goal, not a path constraint.
    """
    mode = target.get('orientation_mode', 'fixed')
    if mode not in ('fixed', 'camera_down_free_yaw'):
        raise AgentError('Unsupported orientation_mode')
    constraint.parameterization = constraint.XYZ_EULER_ANGLES
    constraint.absolute_x_axis_tolerance = 0.08
    constraint.absolute_y_axis_tolerance = 0.08
    constraint.absolute_z_axis_tolerance = math.pi if mode == 'camera_down_free_yaw' else 0.15
    constraint.weight = 1.0


class RosBackend:
    def __init__(self, settings, require_ready=True, qualification_report=None):
        try:
            import rclpy
            from rclpy.action import ActionClient
            from rclpy.executors import SingleThreadedExecutor
            from sensor_msgs.msg import JointState
            from std_srvs.srv import Trigger, SetBool
            from moveit_msgs.srv import GetMotionPlan, ApplyPlanningScene, GetPlanningScene
            from moveit_msgs.action import ExecuteTrajectory
        except ImportError as error:
            raise AgentError('ROS 2 is not sourced. Use scripts/nero_ros2.sh demo --scripted; '
                             'see python -m nero_agent doctor') from error
        self.rclpy, self.settings = rclpy, settings
        self.hardware = settings.mode == 'hardware'
        self.qualification_report = qualification_report
        self.source = 'hardware_feedback' if self.hardware else 'ros2_mock_feedback'
        self.motion_pending = False
        self.goal_handle = None
        self.pending_goal = None
        self.sample = None
        self.sample_received = 0.0
        self.sample_error = None
        self.driver_fault = None
        self.plans = {}
        self.heartbeat_at = 0.0
        self.initial = None
        self.scene_applied = False
        self.context = rclpy.context.Context()
        rclpy.init(context=self.context)
        self.node = rclpy.create_node('nero_agent_' + uuid.uuid4().hex[:8], context=self.context)
        self.callback_executor = None
        ns = settings.namespace
        try:
            self.callback_executor = SingleThreadedExecutor(context=self.context)
            self.callback_executor.add_node(self.node)
            from std_msgs.msg import Empty, String
            self.fault_subscription = self.node.create_subscription(String, ns + '/project/fault', self._fault, 10)
            self.heartbeat = self.node.create_publisher(Empty, ns + '/project/heartbeat', 1)
            topic = ns + ('/feedback/joint_states' if self.hardware else '/joint_states')
            self.subscription = self.node.create_subscription(JointState, topic, self._sample, 10)
            self.info = self.node.create_client(Trigger, ns + '/project/info')
            self.gate = self.node.create_client(SetBool, ns + '/project/control_enable')
            self.estop = self.node.create_client(Trigger, ns + '/project/stop')
            self.abort_status = self.node.create_client(Trigger, ns + '/project/abort_status')
            self.abort_report = self.node.create_client(Trigger, ns + '/project/abort_report')
            self.last_status_poll = 0.0
            self.full_abort_report = None
            self.planner = self.node.create_client(GetMotionPlan, ns + '/plan_kinematic_path')
            self.apply = self.node.create_client(ApplyPlanningScene, ns + '/apply_planning_scene')
            self.scene = self.node.create_client(GetPlanningScene, ns + '/get_planning_scene')
            self.executor = ActionClient(self.node, ExecuteTrajectory, ns + '/execute_trajectory')
            if not require_ready:
                return  # An explicit stop must work even when feedback/MoveIt is unavailable.
            identity = self._call(self.info, Trigger.Request())
            if not identity.success:
                raise AgentError('Backend not ready: ' + identity.message)
            identity = json.loads(identity.message)
            if identity.get('mode') != settings.mode or identity.get('effector') != 'agx_gripper':
                raise AgentError('ROS backend identity does not match config or attached gripper')
            self.initial = self.state()['joints_rad']
        except BaseException:
            self.close()
            raise

    def _fault(self, msg):
        if msg.data:
            self.driver_fault = msg.data

    def _sample(self, msg):
        try:
            if len(msg.name) != len(set(msg.name)) or len(msg.name) != len(msg.position):
                raise AgentError('Duplicate or incomplete joint feedback')
            values = dict(zip(msg.name, msg.position))
            q = vector([values[n] for n in JOINTS])
            if len(msg.velocity) != len(msg.name):
                raise AgentError('Incomplete velocity feedback')
            speeds = dict(zip(msg.name, msg.velocity))
            velocity = vector([speeds[n] for n in JOINTS])
            width = values['gripper']
            if not math.isfinite(width) or not 0 <= width <= 0.1:
                raise AgentError('Invalid measured gripper opening')
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            now = self.node.get_clock().now().nanoseconds * 1e-9
            if not 0 <= now - stamp <= 0.25:
                raise AgentError('Joint/gripper feedback timestamp is stale or invalid')
            self.sample = {'source': self.source, 'joints_rad': list(q), 'velocities_rad_s': list(velocity), 'gripper_width_m': width,
                           'captured_at_unix': stamp}
            self.sample_received = time.monotonic()
            self.sample_error = None
        except (KeyError, TypeError, ValueError, AgentError) as error:
            self.sample_error = str(error)

    def _spin(self):
        if self.hardware and self.motion_pending and time.monotonic() - self.heartbeat_at > 0.1:
            from std_msgs.msg import Empty
            self.heartbeat.publish(Empty())
            self.heartbeat_at = time.monotonic()
        self.callback_executor.spin_once(timeout_sec=0.01)

    def _fresh(self):
        if self.driver_fault:
            raise AgentError('NERO driver stopped: ' + self.driver_fault)
        if self.sample_error or self.sample is None or time.monotonic() - self.sample_received > 0.25:
            raise AgentError(self.sample_error or 'No fresh complete joint/gripper feedback')
        return dict(self.sample)

    def state(self):
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            self._spin()
            try:
                return self._fresh()
            except AgentError:
                pass
        return self._fresh()

    def _wait(self, future, timeout=10.0, monitor=False):
        deadline = time.monotonic() + timeout
        while not future.done():
            self._spin()
            if monitor:
                self._fresh()
            if time.monotonic() >= deadline:
                raise AgentError('ROS request timed out')
        error = future.exception()
        if error is not None:
            raise AgentError(str(error))
        return future.result()

    def _call(self, client, request, timeout=10.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise AgentError('ROS service unavailable: ' + client.srv_name)
        return self._wait(client.call_async(request), timeout)

    def _apply_scene(self):
        from geometry_msgs.msg import Pose
        from shape_msgs.msg import SolidPrimitive
        from moveit_msgs.msg import CollisionObject
        from moveit_msgs.srv import ApplyPlanningScene
        req = ApplyPlanningScene.Request()
        req.scene.is_diff = True
        req.scene.robot_state.is_diff = True
        for spec in self.settings.collision_boxes:
            obj = CollisionObject()
            obj.header.frame_id, obj.id = 'base_link', 'nero_agent_' + spec['id']
            obj.operation = CollisionObject.ADD
            shape = SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[float(x) for x in spec['size_m']])
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = map(float, spec['center_m'])
            pose.orientation.w = 1.0
            obj.primitives, obj.primitive_poses = [shape], [pose]
            req.scene.world.collision_objects.append(obj)
        if not self._call(self.apply, req).success:
            raise AgentError('MoveIt rejected the planning scene')
        self.scene_applied = True

    def _scene_digest(self):
        from moveit_msgs.srv import GetPlanningScene
        from moveit_msgs.msg import PlanningSceneComponents as C
        from rosidl_runtime_py.convert import message_to_ordereddict
        req = GetPlanningScene.Request()
        req.components.components = (C.WORLD_OBJECT_GEOMETRY | C.WORLD_OBJECT_NAMES |
                                     C.ALLOWED_COLLISION_MATRIX | C.ROBOT_STATE_ATTACHED_OBJECTS)
        if self.settings.segmented_execution:
            req.components.components |= C.OCTOMAP
        scene = self._call(self.scene, req).scene
        if self.settings.segmented_execution:
            from .segmented import require_supported_scene
            require_supported_scene(scene, self.settings.collision_boxes, self.preflight_description)
        payload = [message_to_ordereddict(scene.world), message_to_ordereddict(scene.allowed_collision_matrix),
                   [message_to_ordereddict(a) for a in scene.robot_state.attached_collision_objects]]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def plan(self, goal):
        before = self.state()
        if isinstance(goal, dict) and 'position_m' in goal:
            trajectory, digest, summary, goal = self._plan_from_pose(goal, before)
        else:
            goal = vector(goal)
            trajectory, digest, summary = self._plan_from_state(goal, before)
        if self.settings.segmented_execution:
            trajectory, segment_summary = self._prepare_segments(trajectory, before)
            summary.update(segment_summary)
        token = uuid.uuid4().hex
        self.plans[token] = (trajectory, before, goal, digest)
        return {'token': token, 'backend': 'moveit2', 'collision_checked': True,
                'start': before['joints_rad'], 'goal': list(goal), **summary}

    def _plan_from_pose(self, target, before):
        """Ask MoveIt to solve a reviewed Cartesian gripper target."""
        from moveit_msgs.srv import GetMotionPlan
        from moveit_msgs.msg import Constraints, PositionConstraint, OrientationConstraint, MoveItErrorCodes
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Pose
        from rcl_interfaces.srv import GetParameters
        from .trajectory import validate_model_bounds, validate_trajectory
        before_joints = vector(before['joints_rad'])
        client = self.node.create_client(GetParameters, self.settings.namespace + '/move_group/get_parameters')
        try:
            response = self._call(client, GetParameters.Request(names=['robot_description']))
            if len(response.values) != 1 or not response.values[0].string_value:
                raise AgentError('MoveIt robot_description is unavailable')
            validate_model_bounds(response.values[0].string_value, before_joints, before_joints)
            description = response.values[0].string_value
            self.preflight_description = description
        finally:
            self.node.destroy_client(client)
        if not self.scene_applied:
            self._apply_scene()
        digest = self._scene_digest()
        req = GetMotionPlan.Request(); motion = req.motion_plan_request
        motion.group_name = 'arm'; motion.pipeline_id = 'ompl'
        motion.allowed_planning_time = 5.0; motion.num_planning_attempts = 1
        motion.max_velocity_scaling_factor = min(PLANNING_VELOCITY_SCALING,
                                                  self.settings.max_velocity / .08)
        motion.max_acceleration_scaling_factor = min(PLANNING_ACCELERATION_SCALING,
                                                      self.settings.max_acceleration / .15)
        motion.start_state.is_diff = True
        motion.start_state.joint_state.name = list(JOINTS) + ['gripper']
        motion.start_state.joint_state.position = list(before_joints) + [before['gripper_width_m']]
        p = Pose(); p.position.x, p.position.y, p.position.z = map(float, target['position_m'])
        p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = map(float, target['orientation_xyzw'])
        # The vendor MoveIt group exposes tcp_link as its IK tip. The physical
        # gripper links are present in the URDF but are not solver tip links.
        pos = PositionConstraint(); pos.header.frame_id = 'base_link'; pos.link_name = 'tcp_link'
        pos.constraint_region.primitives = [SolidPrimitive(type=SolidPrimitive.BOX, dimensions=[0.004] * 3)]
        pos.constraint_region.primitive_poses = [p]; pos.weight = 1.0
        ori = OrientationConstraint(); ori.header.frame_id = 'base_link'; ori.link_name = 'tcp_link'
        ori.orientation = p.orientation
        configure_goal_orientation(ori, target)
        motion.goal_constraints = [Constraints(position_constraints=[pos], orientation_constraints=[ori])]
        response = self._call(self.planner, req).motion_plan_response
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise AgentError('MoveIt failed to solve Cartesian named pose (error_code=%d)' % response.error_code.val)
        if not response.trajectory.joint_trajectory.points:
            raise AgentError('MoveIt returned an empty Cartesian trajectory')
        trajectory = response.trajectory
        goal = vector(trajectory.joint_trajectory.points[-1].positions)
        summary = validate_trajectory(trajectory.joint_trajectory, before_joints, goal,
                                      self.initial, self.settings)
        summary.update(self._mujoco_preflight(description, trajectory, before))
        return trajectory, digest, summary, goal

    def _mujoco_preflight(self, description, trajectory, before):
        if not self.settings.mujoco_preflight:
            return {}
        try:
            from .mujoco_preflight import validate
        except ImportError as error:
            raise AgentError('Large motion requires MuJoCo preflight: ' + str(error)) from error
        return {'mujoco_preflight': validate(description, trajectory.joint_trajectory,
                                             before, self.initial, self.settings)}

    def _prepare_segments(self, trajectory, before):
        from .segmented import SegmentedRoute, make_segments, STOP_DWELL
        segments = make_segments(trajectory)
        summaries = []
        for segment in segments:
            start = segment.joint_trajectory.points[0].positions
            state = {**before, 'joints_rad': list(start)}
            summaries.append(self._check_segment(self.preflight_description, segment, state))
        duration = sum(s['duration_s'] + STOP_DWELL for s in summaries)
        if duration > self.settings.timeout:
            raise AgentError('Segmented route exceeds total duration budget')
        return SegmentedRoute(segments, self.preflight_description), {
            'execution_mode': 'verified_stop_segments', 'segment_count': len(segments),
            'segment_limit_rad': .10, 'segmented_duration_with_dwells_s': duration,
            'segments': summaries}

    def _check_segment(self, description, trajectory, before):
        from dataclasses import replace
        from .segmented import SEGMENT_TIMEOUT
        from .core import SEGMENT_EXCURSION_RAD
        from .trajectory import validate_trajectory
        from .mujoco_preflight import validate
        points = trajectory.joint_trajectory.points
        settings = replace(self.settings, max_velocity=.02, max_acceleration=.03,
                           timeout=SEGMENT_TIMEOUT)
        if distance(points[0].positions, points[-1].positions) > SEGMENT_EXCURSION_RAD:
            raise AgentError('Segment exceeds per-step excursion allowance')
        summary = validate_trajectory(trajectory.joint_trajectory, before['joints_rad'],
                                      points[-1].positions, self.initial, settings)
        summary['mujoco_preflight'] = validate(description, trajectory.joint_trajectory,
                                              before, self.initial, settings)
        summary['start'] = list(points[0].positions)
        summary['goal'] = list(points[-1].positions)
        return summary

    def _execute_segments(self, route, before, goal, digest):
        from copy import deepcopy
        from .segmented import require_stopped_at, retime_leg
        expected = before['joints_rad']
        deadline = time.monotonic() + self.settings.timeout
        try:
            for index, planned in enumerate(route.segments):
                current = self.state()
                require_stopped_at(current, expected, before['gripper_width_m'])
                if self._scene_digest() != digest:
                    raise AgentError('Planning scene changed between segments; stop and re-plan')
                # Driver is closed here. Align and retime the first endpoint,
                # then check the actual candidate before opening the next gate.
                segment = retime_leg(deepcopy(planned), current['joints_rad'],
                                     planned.joint_trajectory.points[-1].positions)
                summary = self._check_segment(route.description, segment, current)
                if time.monotonic() + summary['duration_s'] + 3 > deadline:
                    raise AgentError('Segmented execution exhausted total time budget')
                token = uuid.uuid4().hex
                endpoint = list(segment.joint_trajectory.points[-1].positions)
                self.plans[token] = (segment, current, endpoint, digest)
                self.executing_segment = True
                try:
                    actual = self.execute({'token': token})
                finally:
                    self.executing_segment = False
                    self.plans.pop(token, None)
                require_stopped_at(actual, endpoint, before['gripper_width_m'])
                expected = endpoint
                callback = getattr(self, 'segment_callback', None)
                if callback:
                    callback({'event': 'segment_completed', 'segment': index+1,
                              'segment_count': len(route.segments), 'state': actual,
                              'plan': summary})
            return actual
        except BaseException:
            # Includes divergence or scene changes while between armed actions.
            self.stop()
            raise

    def _plan_from_state(self, goal, before):
        """Plan from an explicitly supplied fresh measured state."""
        from moveit_msgs.srv import GetMotionPlan
        from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
        from rcl_interfaces.srv import GetParameters
        from .trajectory import validate_model_bounds
        goal = vector(goal)
        before_joints = vector(before['joints_rad'])
        # Read the model actually loaded by MoveIt, not a second local copy.
        client = self.node.create_client(GetParameters, self.settings.namespace + '/move_group/get_parameters')
        try:
            response = self._call(client, GetParameters.Request(names=['robot_description']))
            if len(response.values) != 1 or not response.values[0].string_value:
                raise AgentError('MoveIt robot_description is unavailable for joint-limit validation')
            validate_model_bounds(response.values[0].string_value, before_joints, goal)
            description = response.values[0].string_value
            self.preflight_description = description
        finally:
            self.node.destroy_client(client)
        if distance(goal, self.initial) > self.settings.max_excursion:
            raise AgentError('Target exceeds the captured-start excursion limit: ' +
                             excursion_detail(goal, self.initial, self.settings.max_excursion))
        if not self.scene_applied:
            self._apply_scene()
        digest = self._scene_digest()
        req = GetMotionPlan.Request()
        motion = req.motion_plan_request
        motion.group_name = 'arm'
        motion.pipeline_id = 'ompl'
        motion.allowed_planning_time = 5.0
        motion.num_planning_attempts = 1
        # Launch config sets absolute caps to 0.08 rad/s and 0.15 rad/s².
        motion.max_velocity_scaling_factor = min(PLANNING_VELOCITY_SCALING,
                                                  self.settings.max_velocity / .08)
        motion.max_acceleration_scaling_factor = min(PLANNING_ACCELERATION_SCALING,
                                                      self.settings.max_acceleration / .15)
        motion.start_state.is_diff = True
        motion.start_state.joint_state.name = list(JOINTS) + ['gripper']
        motion.start_state.joint_state.position = list(before_joints) + [before['gripper_width_m']]
        constraints = Constraints()
        constraints.joint_constraints = [JointConstraint(joint_name=n, position=q,
            tolerance_above=0.0005, tolerance_below=0.0005, weight=1.0) for n, q in zip(JOINTS, goal)]
        motion.goal_constraints = [constraints]
        response = self._call(self.planner, req).motion_plan_response
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise AgentError('MoveIt planning failed: error %d' % response.error_code.val)
        trajectory = response.trajectory
        from .trajectory import validate_trajectory
        summary = validate_trajectory(trajectory.joint_trajectory, before_joints, goal,
                                      self.initial, self.settings)
        summary.update(self._mujoco_preflight(description, trajectory, before))
        return trajectory, digest, summary

    def execute(self, plan):
        if self.hardware:
            from .abort_policy import require_verified_controlled_abort
            require_verified_controlled_abort(getattr(self, 'qualification_report', None))
        from std_srvs.srv import SetBool
        from moveit_msgs.action import ExecuteTrajectory
        from moveit_msgs.msg import MoveItErrorCodes
        from action_msgs.msg import GoalStatus
        saved = self.plans.pop(plan.get('token'), None)
        if saved is None:
            raise AgentError('Unknown or already executed plan')
        trajectory, before, goal, digest = saved
        from .segmented import SegmentedRoute, SETUP_DRIFT, STOPPED_SPEED, STOP_DWELL, SEGMENT_TIMEOUT
        if isinstance(trajectory, SegmentedRoute):
            return self._execute_segments(trajectory, before, goal, digest)
        segment = getattr(self, 'executing_segment', False)
        current = self.state()
        if (distance(current['joints_rad'], before['joints_rad']) > (SETUP_DRIFT if segment else .005)
                or abs(current['gripper_width_m'] - before['gripper_width_m']) > 0.001):
            raise AgentError('Arm or gripper changed since planning; re-plan')
        if self._scene_digest() != digest:
            raise AgentError('Planning scene changed since planning; re-plan')
        if not self.executor.wait_for_server(timeout_sec=5.0):
            raise AgentError('MoveIt trajectory execution action unavailable')
        self.motion_pending = True
        try:
            if self.hardware:
                gate = self._call(self.gate, SetBool.Request(data=True))
                if not gate.success:
                    raise AgentError('Hardware execution gate rejected: ' + gate.message)
                # Firmware setup blocks the bridge's timer. Spin for new feedback
                # before entering strict in-flight monitoring or submitting motion.
                self.sample = None
                current = self.state()
                if (distance(current['joints_rad'], before['joints_rad']) > (SETUP_DRIFT if segment else .005)
                        or abs(current['gripper_width_m'] - before['gripper_width_m']) > 0.001):
                    raise AgentError('Arm or gripper changed during hardware setup; re-plan')
                # Driver setup and the final feedback read can happen after
                # the original plan was timed. If the measured state moved,
                # replan from that exact state so MoveIt's first-point
                # velocity/acceleration and controller spline timing match
                # the state that will move.
                if segment and max(map(abs, current['velocities_rad_s'])) > STOPPED_SPEED:
                    raise AgentError('Arm moved during segment setup; stop and re-plan')
                if self.settings.mujoco_preflight and not segment and (
                        distance(current['joints_rad'], before['joints_rad']) > 1e-6
                        or abs(current['gripper_width_m'] - before['gripper_width_m']) > 1e-6):
                    # Do not run an expensive validation with the driver armed,
                    # or replace the trajectory that passed preflight.
                    raise AgentError('State changed during setup; re-plan for MuJoCo preflight')
                if not segment and (distance(current['joints_rad'], before['joints_rad']) > 1e-6
                        or abs(current['gripper_width_m'] - before['gripper_width_m']) > 1e-6):
                    trajectory, digest, _ = self._plan_from_state(goal, current)
            request = ExecuteTrajectory.Goal(trajectory=trajectory)
            self.pending_goal = self.executor.send_goal_async(request)
            self.goal_handle = self._wait(self.pending_goal, monitor=True)
            self.pending_goal = None
            if not self.goal_handle.accepted:
                raise AgentError('MoveIt rejected trajectory execution')
            result = self._wait(self.goal_handle.get_result_async(),
                                (SEGMENT_TIMEOUT if segment else self.settings.timeout) + 5, monitor=True)
            if result.status != GoalStatus.STATUS_SUCCEEDED or result.result.error_code.val != MoveItErrorCodes.SUCCESS:
                raise AgentError('MoveIt execution failed: status %d, error %d' % (result.status, result.result.error_code.val))
            # Action success alone is insufficient: require measured arrival and dwell.
            deadline, stable = time.monotonic() + 3.0, None
            while time.monotonic() < deadline:
                self._spin()
                actual = self._fresh()
                if (distance(actual['joints_rad'], goal) <= self.settings.tolerance
                        and max(map(abs, actual['velocities_rad_s'])) <= (STOPPED_SPEED if segment else .01)):
                    stable = stable or time.monotonic()
                    if time.monotonic() - stable >= (STOP_DWELL if segment else .3):
                        break
                else:
                    stable = None
            else:
                raise AgentError('Controller completed but measured arm did not settle at goal')
            if self.hardware:
                gate = self._call(self.gate, SetBool.Request(data=False))
                if not gate.success:
                    raise AgentError('Could not close hardware execution gate')
            self.motion_pending = False
            self.goal_handle = None
            return actual
        except BaseException:
            self.stop()
            raise

    def commission_abort(self, plan):
        """One bounded trial. The driver triggers and observes the abort independently."""
        from std_srvs.srv import Trigger
        from moveit_msgs.action import ExecuteTrajectory
        from .commissioning import validate_trial_plan
        if not self.hardware:
            raise AgentError('Moving-abort commissioning requires hardware')
        saved = self.plans.pop(plan.get('token'), None)
        if saved is None:
            raise AgentError('Unknown or already consumed commissioning plan')
        trajectory, before, goal, digest = saved
        validate_trial_plan(trajectory.joint_trajectory, before['joints_rad'])
        def check_start():
            current = self.state()
            if (distance(current['joints_rad'], before['joints_rad']) > .0005
                    or max(map(abs, current['velocities_rad_s'])) > .003
                    or abs(current['gripper_width_m'] - before['gripper_width_m']) > .001):
                raise AgentError('Commissioning start changed or is moving; re-plan')
            return current
        check_start()
        if self._scene_digest() != digest:
            raise AgentError('Planning scene changed; re-plan')
        if not self.executor.wait_for_server(timeout_sec=5):
            raise AgentError('MoveIt execution action unavailable')
        gate_client = self.node.create_client(Trigger, self.settings.namespace + '/project/commission_abort')
        self.motion_pending = True
        try:
            gate = self._call(gate_client, Trigger.Request())
            if not gate.success:
                raise AgentError('Commissioning gate rejected: ' + gate.message)
            self.sample = None
            current = check_start()
            self._sync_trajectory_start(trajectory, current, goal)
            self.pending_goal = self.executor.send_goal_async(ExecuteTrajectory.Goal(trajectory=trajectory))
            self.goal_handle = self._wait(self.pending_goal, timeout=2)
            self.pending_goal = None
            if not self.goal_handle.accepted:
                raise AgentError('MoveIt rejected commissioning trajectory')
            completed = self.goal_handle.get_result_async()
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                response = self.poll_abort_status()
                observed = json.loads(response.message)
                self.last_stop_result = observed
                if observed['status'] != 'idle':
                    break
                if completed.done():
                    # An ordinary completion is not a successful moving abort.
                    break
                self._fresh()
                self._spin()
            stopped = self.stop()
            result = stopped['controlled_abort'].get('commissioning', {})
            if result.get('status') in ('pending', 'triggered'):
                result['status'] = 'inconclusive'
                result['reason'] = 'No confirmed moving abort before trajectory completion/deadline'
            return {'status': result.get('status', 'failed'), 'commissioning': result,
                    'controlled_abort': stopped['controlled_abort']}
        except BaseException:
            self.stop()
            raise
        finally:
            self.node.destroy_client(gate_client)

    def poll_abort_status(self):
        from std_srvs.srv import Trigger
        # Continue spinning feedback/heartbeat while limiting requests to 10 Hz.
        deadline = getattr(self, 'last_status_poll', 0.) + .1
        while time.monotonic() < deadline:
            self._spin()
        self.last_status_poll = time.monotonic()
        return self._call(self.abort_status, Trigger.Request(), timeout=1.)

    def fetch_abort_report(self, observed):
        from std_srvs.srv import Trigger
        if (getattr(self, 'full_abort_report', None) is not None
                and self.full_abort_report.get('status') == observed.get('status') == 'failed'):
            self.last_stop_result = self.full_abort_report
            return self.full_abort_report
        response = self._call(self.abort_report, Trigger.Request(), timeout=2.)
        if not response.success:
            raise AgentError('Full abort report unavailable: ' + response.message)
        self.full_abort_report = json.loads(response.message)
        self.last_stop_result = self.full_abort_report
        return self.full_abort_report

    def stop(self):
        from std_srvs.srv import Trigger
        errors = []
        # Driver atomically latches streaming off and requests controller cancellation.
        # Its timer then issues the hold and monitors independently of this client.
        if self.hardware:
            try:
                result = self._call(self.estop, Trigger.Request(), timeout=2)
                if not result.success:
                    try:
                        detail = json.loads(result.message)
                        self.last_stop_result = detail
                        errors.append(detail.get('failure') or detail.get('reason') or 'Driver stop failed')
                    except (ValueError, AttributeError):
                        errors.append(result.message)
            except Exception as error:
                errors.append(str(error))
        if self.pending_goal is not None:
            def cancel_late(future):
                handle = future.result()
                if handle is not None and handle.accepted:
                    handle.cancel_goal_async()
            self.pending_goal.add_done_callback(cancel_late)
        if self.goal_handle is not None and self.goal_handle.accepted:
            try:
                self._wait(self.goal_handle.cancel_goal_async(), timeout=2)
            except Exception as error:
                errors.append(str(error))
        observed = None
        if self.hardware:
            try:
                # Initial settling (5 s), one bounded recheck (5 s), transport margin.
                # Further disturbances never extend this client's total wait.
                deadline = time.monotonic() + 12.0
                while time.monotonic() < deadline:
                    response = self.poll_abort_status()
                    observed = json.loads(response.message)
                    self.last_stop_result = observed
                    if observed['status'] in ('holding', 'failed'):
                        observed = self.fetch_abort_report(observed)
                    if observed['status'] == 'failed':
                        raise AgentError(observed.get('failure') or 'Controlled abort failed')
                    if observed['status'] == 'holding':
                        break
                    self._spin()
                else:
                    raise AgentError('Controlled abort observation timed out')
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise AgentError('Stop could not be confirmed: ' + '; '.join(errors))
        self.motion_pending = False
        self.goal_handle = None
        return {'status': 'holding_observed' if self.hardware else 'stop_requested',
                'source': self.source, 'controlled_abort': observed}

    def _sync_trajectory_start(self, trajectory, current, goal):
        """Align the first controller point with fresh measured feedback."""
        from .trajectory import validate_trajectory
        joint_trajectory = getattr(trajectory, 'joint_trajectory', None)
        # Lightweight backend tests use an opaque trajectory placeholder; the
        # real MoveIt action always supplies a JointTrajectory message.
        if joint_trajectory is None:
            return
        if not hasattr(joint_trajectory, 'points'):
            return
        if not joint_trajectory.points:
            raise AgentError('MoveIt trajectory has no points')
        joint_trajectory.points[0].positions = list(current['joints_rad'])
        try:
            validate_trajectory(joint_trajectory, current['joints_rad'], goal,
                                self.initial, self.settings)
        except Exception as error:
            raise AgentError('Synchronized trajectory start is invalid: ' + str(error))

    def tcp_pose(self, state):
        """FK of the supplied measured snapshot, never controller targets or latest TF."""
        from moveit_msgs.srv import GetPositionFK
        request = GetPositionFK.Request()
        request.header.frame_id = 'base_link'
        request.fk_link_names = ['tcp_link']
        request.robot_state.is_diff = False
        request.robot_state.joint_state.name = list(JOINTS) + ['gripper']
        request.robot_state.joint_state.position = list(vector(state['joints_rad'])) + [state['gripper_width_m']]
        client = self.node.create_client(GetPositionFK, self.settings.namespace + '/compute_fk')
        try:
            response = self._call(client, request)
        finally:
            self.node.destroy_client(client)
        if response.error_code.val != 1 or len(response.pose_stamped) != 1 or list(response.fk_link_names) != ['tcp_link']:
            raise AgentError('Measured-state TCP forward kinematics failed')
        stamped = response.pose_stamped[0]
        if stamped.header.frame_id != 'base_link':
            raise AgentError('TCP forward kinematics returned an unexpected frame')
        p = stamped.pose
        position = [p.position.x, p.position.y, p.position.z]
        orientation = [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]
        if not all(math.isfinite(v) for v in position + orientation):
            raise AgentError('TCP forward kinematics returned nonfinite values')
        return {'frame': 'base_link', 'link': 'tcp_link', 'position_m': position,
                'orientation_xyzw': orientation, 'captured_at_unix': state['captured_at_unix'],
                'source': 'forward kinematics of measured joint feedback'}

    def close(self):
        if self.callback_executor is not None:
            self.callback_executor.shutdown(timeout_sec=1.0)
            self.callback_executor.remove_node(self.node)
            self.callback_executor = None
        self.node.destroy_node()
        if self.context.ok():
            self.context.shutdown()
