"""
Test script: publish MotorCmd to all 12 joints at 20Hz.
Usage: python3 test.py
"""

import rclpy
from rclpy.node import Node
from unitree_legged_msgs.msg import MotorCmd

JOINTS = [
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
]

# Standing pose targets
TARGET = {
    'hip':   0.0,
    'thigh': 1.15,
    'calf':  -2.0,
}

# hip: (-1.0472, 1.0472) - default standing ~0.1
# thigh: (-1.5708, 3.4907) - default standing ~1.15
# calf: (-2.7227, -0.8378) - default standing ~-2.7

KP = 50.0
KD = 2.0
MODE = 10  # PMSM


def _joint_type(name):
    if 'hip' in name:
        return 'hip'
    if 'calf' in name:
        return 'calf'
    return 'thigh'


class TestJointCmd(Node):
    def __init__(self):
        super().__init__('test_joint_cmd')
        self._pubs = {}
        for j in JOINTS:
            self._pubs[j] = self.create_publisher(MotorCmd, f'/go2/{j}/command', 10)
        self._timer = self.create_timer(1.0 / 20.0, self._publish)
        self.get_logger().info(f'Publishing to {len(JOINTS)} joints at 20Hz')

    def _publish(self):
        for j in JOINTS:
            msg = MotorCmd()
            msg.mode = MODE
            msg.q = TARGET[_joint_type(j)]
            msg.dq = 0.0
            msg.tau = 0.0
            msg.kp = KP
            msg.kd = KD
            self._pubs[j].publish(msg)


def main(args=None):
    rclpy.init(args=args)
    rclpy.spin(TestJointCmd())
    rclpy.shutdown()


if __name__ == '__main__':
    main()
