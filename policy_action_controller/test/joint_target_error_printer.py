#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64MultiArray
from unitree_go.msg import LowState


class JointTargetErrorPrinter(Node):
    def __init__(self):
        super().__init__("joint_target_error_printer")
        self.target = None
        self.pos = None
        self.timer = self.create_timer(1.0 / 50.0, self.timer_cb)

        self.create_subscription(
            Float64MultiArray, "/joint_target", self.on_target, 10
        )
        self.create_subscription(
            LowState, "/lowstate", self.on_lowstate, 10
        )
        self.joint_names = [
            "FR_hip_joint",
            "FR_thigh_joint",
            "FR_calf_joint",
            "FL_hip_joint",
            "FL_thigh_joint",
            "FL_calf_joint",
            "RR_hip_joint",
            "RR_thigh_joint",
            "RR_calf_joint",
            "RL_hip_joint",
            "RL_thigh_joint",
            "RL_calf_joint",
        ]

    def on_target(self, msg: Float64MultiArray):
        self.target = list(msg.data)

    def on_lowstate(self, msg: LowState):
        self.pos = [m.q for m in msg.motor_state]

    def timer_cb(self):
        if self.target is None or self.pos is None:
            return
        n = min(len(self.target), len(self.pos), 12)
        if n == 0:
            return

        lines = []
        lines.append(" idx  joint              target        pos         error")
        for i in range(n):
            t = self.target[i]
            p = self.pos[i]
            e = t - p
            name = self.joint_names[i] if i < len(self.joint_names) else f"joint_{i}"
            lines.append(f" {i:2d}  {name:16}  {t:10.4f}  {p:10.4f}  {e:10.4f}")
        print("\n".join(lines))


def main():
    rclpy.init()
    node = JointTargetErrorPrinter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
