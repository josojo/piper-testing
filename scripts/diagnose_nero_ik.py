"""Offline MoveIt IK/collision diagnosis inside the ROS image; no execution calls."""
import argparse
import json
import math
from pathlib import Path
import random
import signal
import subprocess
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='nero-agent.local.json')
    parser.add_argument('--report', default='reports/photo-staging.json')
    parser.add_argument('--pose', default='table_photo_center')
    parser.add_argument('--output', default='reports/table-photo-center-ik.json')
    parser.add_argument('--pose-checks', action='store_true',
                        help='Check FK-to-IK recovery and target position with captured orientation')
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    recorded = json.loads(Path(args.report).read_text())
    initial = next(e for e in recorded['events'] if e['event'] == 'initial_state')
    state = initial['state']
    # Replay the failed target, even if the local configuration has since changed.
    target = initial['named_poses'][args.pose]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.resolve() in (Path(args.config).resolve(), Path(args.report).resolve()):
        raise ValueError('Output must not overwrite an input')
    import rclpy
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import CollisionObject, RobotState
    from moveit_msgs.srv import ApplyPlanningScene, GetPositionIK, GetStateValidity, GetPositionFK
    from rcl_interfaces.srv import GetParameters
    from shape_msgs.msg import SolidPrimitive
    from sensor_msgs.msg import JointState

    rclpy.init()
    node = rclpy.create_node('offline_ik_diagnostic')
    namespace = '/nero_ik_diag'
    joints = ['joint%d' % i for i in range(1, 8)]
    result = {'mode': 'offline_mock', 'motion_commands_sent': False,
              'source_report': args.report, 'target': target, 'captured_state': state,
              'collision_boxes': config['collision_boxes'], 'attempts': []}

    def call(service, name, request):
        client = node.create_client(service, namespace + name)
        try:
            if not client.wait_for_service(timeout_sec=30):
                raise RuntimeError('Service unavailable: ' + name)
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=10)
            if not future.done():
                raise RuntimeError('Service timed out: ' + name)
            return future.result()
        finally:
            node.destroy_client(client)

    def robot_state(q):
        return RobotState(joint_state=JointState(
            name=joints + ['gripper'], position=list(q) + [state['gripper_width_m']]),
            is_diff=False)

    def validity(rs):
        reply = call(GetStateValidity, '/check_state_validity',
                     GetStateValidity.Request(robot_state=rs, group_name='arm'))
        return {'valid': reply.valid, 'contacts': [
            {'body_1': c.contact_body_1, 'body_2': c.contact_body_2, 'depth_m': c.depth}
            for c in reply.contacts]}

    log_path = output.with_suffix('.ros.log')
    with log_path.open('w') as log:
        process = subprocess.Popen(['ros2', 'launch', '/work/ros2/nero.launch.py',
                                    'mode:=mock', 'namespace:=nero_ik_diag'],
                                   stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            parameters = call(GetParameters, '/move_group/get_parameters',
                              GetParameters.Request(names=['robot_description']))
            root = ET.fromstring(parameters.values[0].string_value)
            bounds = []
            for name in joints:
                limit = root.find('./joint[@name="%s"]/limit' % name)
                bounds.append((float(limit.get('lower')), float(limit.get('upper'))))
            request = ApplyPlanningScene.Request()
            request.scene.is_diff = True
            request.scene.robot_state = robot_state(state['joints_rad'])
            for box in config['collision_boxes']:
                obj = CollisionObject(id=box['id'], operation=CollisionObject.ADD)
                obj.header.frame_id = 'base_link'
                pose = Pose()
                pose.position.x, pose.position.y, pose.position.z = map(float, box['center_m'])
                pose.orientation.w = 1.0
                obj.primitives = [SolidPrimitive(type=SolidPrimitive.BOX,
                                                dimensions=list(map(float, box['size_m'])))]
                obj.primitive_poses = [pose]
                request.scene.world.collision_objects.append(obj)
            if not call(ApplyPlanningScene, '/apply_planning_scene', request).success:
                raise RuntimeError('Planning scene update failed')
            result['start_validity'] = validity(robot_state(state['joints_rad']))
            cases = {'requested_pose': target}
            if args.pose_checks:
                fk = GetPositionFK.Request(robot_state=robot_state(state['joints_rad']),
                                           fk_link_names=['tcp_link'])
                fk.header.frame_id = 'base_link'
                answer = call(GetPositionFK, '/compute_fk', fk)
                if answer.error_code.val != 1 or len(answer.pose_stamped) != 1:
                    raise RuntimeError('Captured-state FK failed: %d' % answer.error_code.val)
                p = answer.pose_stamped[0].pose
                captured_pose = {
                    'frame': 'base_link',
                    'position_m': [p.position.x, p.position.y, p.position.z],
                    'orientation_xyzw': [p.orientation.x, p.orientation.y,
                                         p.orientation.z, p.orientation.w]}
                result['captured_tcp_pose'] = captured_pose
                cases = {
                    'known_reachable': captured_pose,
                    'target_position_captured_orientation': {
                        **target, 'orientation_xyzw': captured_pose['orientation_xyzw']}}
            result['cases'] = cases
            rng = random.Random(0)
            seeds = [state['joints_rad']] + [[rng.uniform(lo, hi) for lo, hi in bounds]
                                           for _ in range(15)]
            for case_name, case_target in cases.items():
                # First seed always reproduces the captured joints. The known
                # pose test uses that exact seed to test basic solver recovery.
                case_seeds = seeds[:1] if case_name == 'known_reachable' else seeds
                for avoid_collisions, index, seed in (
                        (avoid, i, q) for avoid in (False, True)
                        for i, q in enumerate(case_seeds)):
                    req = GetPositionIK.Request()
                    ik = req.ik_request
                    ik.group_name = 'arm'
                    ik.ik_link_name = 'tcp_link'
                    ik.robot_state = robot_state(seed)
                    ik.avoid_collisions = avoid_collisions
                    ik.timeout.sec = 1
                    ik.pose_stamped.header.frame_id = 'base_link'
                    pose = ik.pose_stamped.pose
                    pose.position.x, pose.position.y, pose.position.z = map(float, case_target['position_m'])
                    (pose.orientation.x, pose.orientation.y, pose.orientation.z,
                     pose.orientation.w) = map(float, case_target['orientation_xyzw'])
                    reply = call(GetPositionIK, '/compute_ik', req)
                    attempt = {'case': case_name, 'avoid_collisions': avoid_collisions, 'seed_index': index,
                               'error_code': reply.error_code.val}
                    if reply.error_code.val == 1:
                        values = dict(zip(reply.solution.joint_state.name,
                                          reply.solution.joint_state.position))
                        q = [values[name] for name in joints]
                        solved = robot_state(q)
                        attempt['joints_rad'] = q
                        attempt['validity'] = validity(solved)
                        attempt['excursions_rad'] = [abs(a-b) for a, b in zip(q, state['joints_rad'])]
                        attempt['peak_excursion_rad'] = max(attempt['excursions_rad'])
                        fk = GetPositionFK.Request(robot_state=solved, fk_link_names=['tcp_link'])
                        fk.header.frame_id = 'base_link'
                        answer = call(GetPositionFK, '/compute_fk', fk)
                        attempt['fk_error_code'] = answer.error_code.val
                        if answer.error_code.val == 1:
                            p = answer.pose_stamped[0].pose
                            attempt['fk_position_m'] = [p.position.x, p.position.y, p.position.z]
                            attempt['fk_orientation_xyzw'] = [p.orientation.x, p.orientation.y,
                                                              p.orientation.z, p.orientation.w]
                            attempt['position_error_m'] = math.dist(
                                attempt['fk_position_m'], case_target['position_m'])
                            actual, desired = attempt['fk_orientation_xyzw'], case_target['orientation_xyzw']
                            dot = sum(a*b for a, b in zip(actual, desired))
                            norm = math.sqrt(sum(a*a for a in actual) * sum(b*b for b in desired))
                            attempt['orientation_error_rad'] = 2 * math.acos(min(1., abs(dot / norm)))
                    result['attempts'].append(attempt)
                    print(json.dumps(attempt), flush=True)
            result['summary'] = {name: {
                'without_collision_check': sum(a['error_code'] == 1 for a in result['attempts']
                                               if a['case'] == name and not a['avoid_collisions']),
                'with_collision_check': sum(a['error_code'] == 1 for a in result['attempts']
                                            if a['case'] == name and a['avoid_collisions']),
                'attempts_per_mode': 1 if name == 'known_reachable' else len(seeds)}
                for name in cases}
            result['limitations'] = ('Exact-pose IK with finite seed search; failure does not prove '
                                    'unreachability. Uses saved state and current configured boxes, '
                                    'not live hardware. No path planning or execution qualification.')
        except Exception as error:
            result['error'] = str(error)
            raise
        finally:
            output.write_text(json.dumps(result, indent=2) + '\n')
            # Stop only the isolated launch process created above.
            if process.poll() is None:
                process.send_signal(signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            node.destroy_node()
            rclpy.shutdown()
    print('Report: ' + str(output), flush=True)


if __name__ == '__main__':
    main()
