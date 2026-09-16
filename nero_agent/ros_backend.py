"""MoveIt services/actions with independent measured-state completion checks.

ROS imports stay here so the offline contract test works with ordinary Python.
"""
import hashlib
import json
import math
import time
import uuid

from .core import AgentError, JOINTS, distance, vector


class RosBackend:
    def __init__(self, settings, require_ready=True):
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
        scene = self._call(self.scene, req).scene
        payload = [message_to_ordereddict(scene.world), message_to_ordereddict(scene.allowed_collision_matrix),
                   [message_to_ordereddict(a) for a in scene.robot_state.attached_collision_objects]]
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def plan(self, goal):
        from moveit_msgs.srv import GetMotionPlan
        from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes
        from rcl_interfaces.srv import GetParameters
        from .trajectory import validate_model_bounds
        before = self.state()
        goal = vector(goal)
        # Read the model actually loaded by MoveIt, not a second local copy.
        client = self.node.create_client(GetParameters, self.settings.namespace + '/move_group/get_parameters')
        try:
            response = self._call(client, GetParameters.Request(names=['robot_description']))
            if len(response.values) != 1 or not response.values[0].string_value:
                raise AgentError('MoveIt robot_description is unavailable for joint-limit validation')
            validate_model_bounds(response.values[0].string_value, before['joints_rad'], goal)
        finally:
            self.node.destroy_client(client)
        if distance(goal, self.initial) > self.settings.max_excursion:
            raise AgentError('Target exceeds the captured-start excursion limit')
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
        motion.max_velocity_scaling_factor = 1.0
        motion.max_acceleration_scaling_factor = 1.0
        motion.start_state.is_diff = True
        motion.start_state.joint_state.name = list(JOINTS) + ['gripper']
        motion.start_state.joint_state.position = list(before['joints_rad']) + [before['gripper_width_m']]
        constraints = Constraints()
        constraints.joint_constraints = [JointConstraint(joint_name=n, position=q,
            tolerance_above=0.0005, tolerance_below=0.0005, weight=1.0) for n, q in zip(JOINTS, goal)]
        motion.goal_constraints = [constraints]
        response = self._call(self.planner, req).motion_plan_response
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            raise AgentError('MoveIt planning failed: error %d' % response.error_code.val)
        trajectory = response.trajectory
        from .trajectory import validate_trajectory
        summary = validate_trajectory(trajectory.joint_trajectory, before['joints_rad'], goal,
                                      self.initial, self.settings)
        token = uuid.uuid4().hex
        self.plans[token] = (trajectory, before, goal, digest)
        return {'token': token, 'backend': 'moveit2', 'collision_checked': True,
                'start': before['joints_rad'], 'goal': list(goal), **summary}

    def execute(self, plan):
        from std_srvs.srv import SetBool
        from moveit_msgs.action import ExecuteTrajectory
        from moveit_msgs.msg import MoveItErrorCodes
        from action_msgs.msg import GoalStatus
        saved = self.plans.pop(plan.get('token'), None)
        if saved is None:
            raise AgentError('Unknown or already executed plan')
        trajectory, before, goal, digest = saved
        current = self.state()
        if (distance(current['joints_rad'], before['joints_rad']) > 0.005
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
                if (distance(current['joints_rad'], before['joints_rad']) > 0.005
                        or abs(current['gripper_width_m'] - before['gripper_width_m']) > 0.001):
                    raise AgentError('Arm or gripper changed during hardware setup; re-plan')
            request = ExecuteTrajectory.Goal(trajectory=trajectory)
            self.pending_goal = self.executor.send_goal_async(request)
            self.goal_handle = self._wait(self.pending_goal, monitor=True)
            self.pending_goal = None
            if not self.goal_handle.accepted:
                raise AgentError('MoveIt rejected trajectory execution')
            result = self._wait(self.goal_handle.get_result_async(), self.settings.timeout + 5, monitor=True)
            if result.status != GoalStatus.STATUS_SUCCEEDED or result.result.error_code.val != MoveItErrorCodes.SUCCESS:
                raise AgentError('MoveIt execution failed: status %d, error %d' % (result.status, result.result.error_code.val))
            # Action success alone is insufficient: require measured arrival and dwell.
            deadline, stable = time.monotonic() + 3.0, None
            while time.monotonic() < deadline:
                self._spin()
                actual = self._fresh()
                if (distance(actual['joints_rad'], goal) <= self.settings.tolerance
                        and max(map(abs, actual['velocities_rad_s'])) <= 0.01):
                    stable = stable or time.monotonic()
                    if time.monotonic() - stable >= 0.3:
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

    def stop(self):
        from std_srvs.srv import Trigger
        errors = []
        # Hardware stop is attempted before ROS cancellation can block.
        if self.hardware:
            try:
                result = self._call(self.estop, Trigger.Request(), timeout=2)
                if not result.success:
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
        if errors:
            raise AgentError('Stop could not be confirmed: ' + '; '.join(errors))
        self.motion_pending = False
        self.goal_handle = None
        return {'status': 'stop_requested', 'source': self.source}

    def close(self):
        if self.callback_executor is not None:
            self.callback_executor.shutdown(timeout_sec=1.0)
            self.callback_executor.remove_node(self.node)
            self.callback_executor = None
        self.node.destroy_node()
        if self.context.ok():
            self.context.shutdown()
