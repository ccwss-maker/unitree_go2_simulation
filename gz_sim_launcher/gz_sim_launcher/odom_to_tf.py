#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

class OdomToTF(Node):
    def __init__(self):
        super().__init__('odom_to_tf')
        self.tf_broadcaster = TransformBroadcaster(self)
        self.last_msg = None

        # 订阅 odometry
        self.create_subscription(Odometry, '/odom', self.odom_callback, 10)

        # 创建 200Hz 定时器广播 TF
        timer_period = 1.0 / 200.0  # seconds
        self.create_timer(timer_period, self.timer_callback)

    def odom_callback(self, msg):
        self.last_msg = msg

    def timer_callback(self):
        if self.last_msg is None:
            return
        msg = self.last_msg
        t = TransformStamped()
        t.header = msg.header
        t.child_frame_id = msg.child_frame_id
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        t.transform.rotation = msg.pose.pose.orientation
        self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = OdomToTF()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
