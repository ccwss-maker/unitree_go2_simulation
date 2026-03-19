#!/usr/bin/env python3
from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from unitree_legged_msgs.msg import MotorCmd

PMSM = 0x0A


def _normalize_topic_prefix(topic_prefix: str) -> str:
    normalized = str(topic_prefix).strip()
    if not normalized:
        return ""
    return normalized.rstrip("/")


class SimCmdController:
    def __init__(
        self,
        node: Node,
        *,
        cmd_hz: float,
        target_joint_angles: list[float],
        sim_order_indices: list[int],
        joint_names: list[str],
        kp: float,
        kd: float,
        joint_command_prefix: str,
    ) -> None:
        if len(target_joint_angles) != 12:
            raise ValueError("target_joint_angles must contain 12 values")
        if len(sim_order_indices) != 12:
            raise ValueError("sim_order_indices must contain 12 indices")
        if len(joint_names) != 12:
            raise ValueError("joint_names must contain 12 values")
        if cmd_hz <= 0.0:
            raise ValueError("cmd_hz must be positive")

        self.target_joint_angles = [float(v) for v in target_joint_angles]
        self.sim_order_indices = [int(v) for v in sim_order_indices]
        self.joint_names = [str(name) for name in joint_names]
        self.kp = float(kp)
        self.kd = float(kd)
        self.active = False
        self.cmd_period = 1.0 / cmd_hz
        self.joint_command_prefix = _normalize_topic_prefix(joint_command_prefix)

        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )
        self.publishers = {}
        for joint_name in self.joint_names:
            topic_name = f"{self.joint_command_prefix}/{joint_name}/command"
            self.publishers[joint_name] = node.create_publisher(MotorCmd, topic_name, qos_cmd)

        self.timer = node.create_timer(self.cmd_period, self._publish)

    def enable(self) -> None:
        self.active = True

    def disable(self) -> None:
        self.active = False

    def set_target_joint_angles(self, target_joint_angles: list[float]) -> None:
        if len(target_joint_angles) != 12:
            raise ValueError("target_joint_angles must contain 12 values")
        self.target_joint_angles = [float(v) for v in target_joint_angles]

    def set_gains(self, kp: float, kd: float) -> None:
        self.kp = float(kp)
        self.kd = float(kd)

    def execute_joint_transition(
        self,
        current_joint_angles: list[float],
        target_joint_angles: list[float],
        duration_sec: float,
    ) -> bool:
        if len(current_joint_angles) != 12:
            return False
        if len(target_joint_angles) != 12:
            return False
        if duration_sec <= 0.0:
            return False

        was_active = self.active
        self.active = False

        start_joint_angles = [float(v) for v in current_joint_angles]
        final_joint_angles = [float(v) for v in target_joint_angles]
        start_time = time.monotonic()

        while rclpy.ok():
            elapsed = time.monotonic() - start_time
            alpha = min(1.0, elapsed / duration_sec)
            interp_joint_angles = [
                start + alpha * (target - start)
                for start, target in zip(start_joint_angles, final_joint_angles)
            ]
            self._publish_target(interp_joint_angles)

            if alpha >= 1.0:
                self.target_joint_angles = final_joint_angles
                self.active = was_active
                return True

            time.sleep(self.cmd_period)

        self.active = was_active
        return False

    def _publish_target(self, target_joint_angles: list[float]) -> None:
        for sim_idx in range(12):
            joint_idx = self.sim_order_indices[sim_idx]
            if joint_idx < 0 or joint_idx >= 12:
                raise ValueError("sim_order_indices must contain indices in [0, 11]")
            joint_name = self.joint_names[sim_idx]

            msg = MotorCmd()
            msg.mode = PMSM
            msg.q = float(target_joint_angles[joint_idx])
            msg.dq = 0.0
            msg.tau = 0.0
            msg.kp = self.kp
            msg.kd = self.kd
            self.publishers[joint_name].publish(msg)

    def _publish(self) -> None:
        if not self.active:
            return
        self._publish_target(self.target_joint_angles)
