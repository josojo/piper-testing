"""Explicit identity for a ROS mock stack; never imports a hardware SDK."""
import argparse
import json


def main():
    import rclpy
    from std_srvs.srv import Trigger
    parser = argparse.ArgumentParser()
    parser.add_argument('--namespace', default='/nero')
    args = parser.parse_args()
    rclpy.init()
    node = rclpy.create_node('nero_mock_identity', namespace=args.namespace)
    def info(request, response):
        response.success = True
        response.message = json.dumps({'mode': 'mock', 'effector': 'agx_gripper'})
        return response
    node.create_service(Trigger, 'project/info', info)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
