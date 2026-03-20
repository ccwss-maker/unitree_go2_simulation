#!/usr/bin/env python3
from __future__ import annotations

import math
import time
from collections.abc import Callable
from threading import Lock

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu, JointState
from unitree_legged_msgs.msg import MotorCmd

PMSM = 0x0A


def _normalize_topic_prefix(topic_prefix: str) -> str:
    normalized = str(topic_prefix).strip()
    if not normalized:
        return ''
    return normalized.rstrip('/')


class Sim2SimStateAdapter:
    def __init__(self, node: Node, control_cfg: dict[str, object]) -> None:
        self.node = node
        self.control_cfg = control_cfg
        self._state_lock = Lock()
        self._current_joint_angles: list[float] | None = None
        self._current_joint_velocities: list[float] | None = None
        self._base_linear_velocity = [0.0] * 3
        self._base_angular_velocity = [0.0] * 3
        self._rpy = [0.0] * 3
        self._cmd_vel = [0.0] * 3
        self._has_imu = False
        self._has_odom = False
        self._has_cmd_vel = False
        self._has_motion_state = False
        self._last_missing_joint_warning_time = 0.0

        topics_cfg = control_cfg['topics']['sim2sim']
        self.joint_source_topic = str(topics_cfg['joint_states'])
        self.wait_message = (
            f"Waiting for {topics_cfg['joint_states']}, "
            f"{topics_cfg['odom']}, "
            f"{topics_cfg['imu']} and "
            f"{control_cfg['topics']['cmd_vel']} before policy inference"
        )

        self.joint_states_sub = node.create_subscription(
            JointState,
            self.joint_source_topic,
            self._on_joint_states,
            10,
        )
        self.odom_sub = node.create_subscription(
            Odometry,
            str(topics_cfg['odom']),
            self._on_odom,
            10,
        )
        self.imu_sub = node.create_subscription(
            Imu,
            str(topics_cfg['imu']),
            self._on_imu,
            10,
        )
        self.cmd_vel_sub = node.create_subscription(
            Twist,
            str(control_cfg['topics']['cmd_vel']),
            self._on_cmd_vel,
            10,
        )

    def get_current_joint_angles(self) -> list[float] | None:
        with self._state_lock:
            if self._current_joint_angles is None:
                return None
            return list(self._current_joint_angles)

    def wait_for_initial_joint_angles(
        self,
        timeout_sec: float,
        stop_requested_cb: Callable[[], bool],
    ) -> list[float] | None:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not stop_requested_cb():
            current_joint_angles = self.get_current_joint_angles()
            if current_joint_angles is not None:
                return current_joint_angles
            if time.monotonic() >= deadline:
                break
            rclpy.spin_once(self.node, timeout_sec=0.05)
        return self.get_current_joint_angles()

    def get_policy_state_snapshot(self) -> dict[str, list[float]] | None:
        with self._state_lock:
            if self._current_joint_angles is None or self._current_joint_velocities is None:
                return None
            if not self._has_motion_state or not self._has_cmd_vel:
                return None

            return {
                'current_joint_angles': list(self._current_joint_angles),
                'current_joint_velocities': list(self._current_joint_velocities),
                'base_linear_velocity': list(self._base_linear_velocity),
                'base_angular_velocity': list(self._base_angular_velocity),
                'rpy': list(self._rpy),
                'cmd_vel': list(self._cmd_vel),
            }

    def _on_joint_states(self, msg: JointState) -> None:
        joint_name_to_idx = {name: idx for idx, name in enumerate(msg.name)}
        current_joint_angles = [0.0] * 12
        current_joint_velocities = [0.0] * 12
        missing_joint_names: list[str] = []

        for sim_idx, joint_name in enumerate(self.control_cfg['joint_names']):
            joint_idx = self.control_cfg['sim_order_indices'][sim_idx]
            if joint_idx < 0 or joint_idx >= 12:
                missing_joint_names.append(f'invalid_index:{joint_idx}')
                continue
            msg_idx = joint_name_to_idx.get(joint_name)
            if msg_idx is None or msg_idx >= len(msg.position):
                missing_joint_names.append(joint_name)
                continue
            current_joint_angles[joint_idx] = float(msg.position[msg_idx])
            if msg_idx < len(msg.velocity):
                current_joint_velocities[joint_idx] = float(msg.velocity[msg_idx])

        if missing_joint_names:
            now = time.monotonic()
            if now - self._last_missing_joint_warning_time >= 1.0:
                self.node.get_logger().warning(
                    f"Missing joints in joint_states: {', '.join(missing_joint_names)}"
                )
                self._last_missing_joint_warning_time = now
            return

        with self._state_lock:
            self._current_joint_angles = current_joint_angles
            self._current_joint_velocities = current_joint_velocities

    def _on_odom(self, msg: Odometry) -> None:
        with self._state_lock:
            self._base_linear_velocity = [
                float(msg.twist.twist.linear.x),
                float(msg.twist.twist.linear.y),
                float(msg.twist.twist.linear.z),
            ]
            self._base_angular_velocity = [
                float(msg.twist.twist.angular.x),
                float(msg.twist.twist.angular.y),
                float(msg.twist.twist.angular.z),
            ]
            self._has_odom = True
            self._has_motion_state = self._has_odom and self._has_imu

    def _on_imu(self, msg: Imu) -> None:
        roll, pitch, yaw = self._quat_to_rpy(
            float(msg.orientation.x),
            float(msg.orientation.y),
            float(msg.orientation.z),
            float(msg.orientation.w),
        )
        with self._state_lock:
            self._rpy = [roll, pitch, yaw]
            self._has_imu = True
            self._has_motion_state = self._has_odom and self._has_imu

    def _on_cmd_vel(self, msg: Twist) -> None:
        with self._state_lock:
            self._cmd_vel = [
                float(msg.linear.x),
                float(msg.linear.y),
                float(msg.angular.z),
            ]
            self._has_cmd_vel = True

    @staticmethod
    def _quat_to_rpy(x: float, y: float, z: float, w: float) -> tuple[float, float, float]:
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw


class Sim2SimControlBackend:
    def __init__(self, node: Node, control_cfg: dict[str, object]) -> None:
        if len(control_cfg['default_joint_angles']) != 12:
            raise ValueError('default_joint_angles must contain 12 values')
        if len(control_cfg['sim_order_indices']) != 12:
            raise ValueError('sim_order_indices must contain 12 indices')
        if len(control_cfg['joint_names']) != 12:
            raise ValueError('joint_names must contain 12 values')

        cmd_hz = float(control_cfg['motor']['cmd_hz'])
        if cmd_hz <= 0.0:
            raise ValueError('motor.cmd_hz must be positive')

        self.node = node
        self.target_joint_angles = [float(v) for v in control_cfg['default_joint_angles']]
        self.sim_order_indices = [int(v) for v in control_cfg['sim_order_indices']]
        self.joint_names = [str(name) for name in control_cfg['joint_names']]
        self.kp = float(control_cfg['stand']['kp'])
        self.kd = float(control_cfg['stand']['kd'])
        self.stand_kp = float(control_cfg['stand']['kp'])
        self.stand_kd = float(control_cfg['stand']['kd'])
        self.active = False
        self.cmd_period = 1.0 / cmd_hz
        self.joint_command_prefix = _normalize_topic_prefix(
            str(control_cfg['topics']['sim2sim']['joint_command_prefix'])
        )

        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )
        self.publishers = {}
        for joint_name in self.joint_names:
            topic_name = f'{self.joint_command_prefix}/{joint_name}/command'
            self.publishers[joint_name] = node.create_publisher(MotorCmd, topic_name, qos_cmd)

        self.timer = node.create_timer(self.cmd_period, self._publish)

    def wait_until_ready(self, stop_requested_cb: Callable[[], bool]) -> bool:
        del stop_requested_cb
        return True

    def enter_control_mode(
        self,
        stop_requested_cb: Callable[[], bool],
    ) -> tuple[bool, str]:
        del stop_requested_cb
        return True, 'sim2sim mode: skipped motion mode initialize'

    def restore_default_mode(
        self,
        current_joint_angles: list[float] | None,
        target_joint_angles: list[float],
        duration_sec: float,
        stop_requested_cb: Callable[[], bool],
        *,
        ignore_stop_requested: bool = False,
    ) -> tuple[bool, str]:
        del stop_requested_cb
        del ignore_stop_requested

        if current_joint_angles is None:
            self.disable()
            return False, 'Failed to read current joint angles for sim2sim stand restore'

        self.set_gains(self.stand_kp, self.stand_kd)
        success = self.execute_joint_transition(
            current_joint_angles=current_joint_angles,
            target_joint_angles=target_joint_angles,
            duration_sec=duration_sec,
        )
        self.disable()
        if success:
            return True, 'sim2sim mode: restored standing pose'
        return False, 'Failed to restore standing pose in sim2sim mode'

    def enable(self) -> None:
        self.active = True

    def disable(self) -> None:
        self.active = False

    def set_target_joint_angles(self, target_joint_angles: list[float]) -> None:
        if len(target_joint_angles) != 12:
            raise ValueError('target_joint_angles must contain 12 values')
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
                raise ValueError('sim_order_indices must contain indices in [0, 11]')
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
