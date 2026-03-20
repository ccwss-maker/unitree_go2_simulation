#!/usr/bin/env python3
from __future__ import annotations

import time
from collections.abc import Callable
from threading import Lock

import rclpy
from geometry_msgs.msg import Twist
from lowcmd_controller import LowCmdController
from rclpy.node import Node
from std_srvs.srv import Trigger
from unitree_go.msg import LowState, SportModeState


class Sim2RealStateAdapter:
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
        self._has_motion_state = False
        self._has_cmd_vel = False

        topics_cfg = control_cfg['topics']['sim2real']
        self.joint_source_topic = str(topics_cfg['lowstate'])
        self.wait_message = (
            f"Waiting for {topics_cfg['lowstate']}, "
            f"{topics_cfg['sportstate']} and "
            f"{control_cfg['topics']['cmd_vel']} before policy inference"
        )

        self.lowstate_sub = node.create_subscription(
            LowState,
            self.joint_source_topic,
            self._on_lowstate,
            10,
        )
        self.sportstate_sub = node.create_subscription(
            SportModeState,
            str(topics_cfg['sportstate']),
            self._on_sportstate,
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

    def _on_lowstate(self, msg: LowState) -> None:
        current_joint_angles = [0.0] * 12
        current_joint_velocities = [0.0] * 12
        for real_idx in range(12):
            base_idx = self.control_cfg['real_order_indices'][real_idx]
            current_joint_angles[base_idx] = float(msg.motor_state[real_idx].q)
            current_joint_velocities[base_idx] = float(msg.motor_state[real_idx].dq)

        with self._state_lock:
            self._current_joint_angles = current_joint_angles
            self._current_joint_velocities = current_joint_velocities

    def _on_sportstate(self, msg: SportModeState) -> None:
        with self._state_lock:
            self._base_linear_velocity = [float(value) for value in msg.velocity[0:3]]
            self._base_angular_velocity = [
                float(value) for value in msg.imu_state.gyroscope[0:3]
            ]
            self._rpy = [float(value) for value in msg.imu_state.rpy[0:3]]
            self._has_motion_state = True

    def _on_cmd_vel(self, msg: Twist) -> None:
        with self._state_lock:
            self._cmd_vel = [
                float(msg.linear.x),
                float(msg.linear.y),
                float(msg.angular.z),
            ]
            self._has_cmd_vel = True


class Sim2RealControlBackend:
    def __init__(self, node: Node, control_cfg: dict[str, object]) -> None:
        self.node = node
        self.controller = LowCmdController(
            node,
            cmd_hz=float(control_cfg['motor']['cmd_hz']),
            target_joint_angles=control_cfg['default_joint_angles'],
            real_order_indices=control_cfg['real_order_indices'],
            kp=float(control_cfg['stand']['kp']),
            kd=float(control_cfg['stand']['kd']),
        )
        self.initialize_client = node.create_client(Trigger, '/motion_mode_manager/initialize')
        self.exit_client = node.create_client(Trigger, '/motion_mode_manager/exit')

    def wait_until_ready(self, stop_requested_cb: Callable[[], bool]) -> bool:
        while rclpy.ok() and not stop_requested_cb():
            if self.initialize_client.wait_for_service(timeout_sec=1.0):
                break
            self.node.get_logger().info('Waiting for /motion_mode_manager/initialize ...')

        while rclpy.ok() and not stop_requested_cb():
            if self.exit_client.wait_for_service(timeout_sec=1.0):
                return True
            self.node.get_logger().info('Waiting for /motion_mode_manager/exit ...')

        return False

    def enter_control_mode(
        self,
        stop_requested_cb: Callable[[], bool],
    ) -> tuple[bool, str]:
        return self._call_trigger(
            self.initialize_client,
            'initialize',
            stop_requested_cb,
        )

    def restore_default_mode(
        self,
        current_joint_angles: list[float] | None,
        target_joint_angles: list[float],
        duration_sec: float,
        stop_requested_cb: Callable[[], bool],
        *,
        ignore_stop_requested: bool = False,
    ) -> tuple[bool, str]:
        del current_joint_angles
        del target_joint_angles
        del duration_sec
        self.disable()
        return self._call_trigger(
            self.exit_client,
            'exit',
            stop_requested_cb,
            ignore_stop_requested=ignore_stop_requested,
        )

    def enable(self) -> None:
        self.controller.enable()

    def disable(self) -> None:
        self.controller.disable()

    def set_target_joint_angles(self, target_joint_angles: list[float]) -> None:
        self.controller.set_target_joint_angles(target_joint_angles)

    def set_gains(self, kp: float, kd: float) -> None:
        self.controller.set_gains(kp, kd)

    def execute_joint_transition(
        self,
        current_joint_angles: list[float],
        target_joint_angles: list[float],
        duration_sec: float,
    ) -> bool:
        return self.controller.execute_joint_transition(
            current_joint_angles=current_joint_angles,
            target_joint_angles=target_joint_angles,
            duration_sec=duration_sec,
        )

    def _call_trigger(
        self,
        client,
        action_name: str,
        stop_requested_cb: Callable[[], bool],
        *,
        ignore_stop_requested: bool = False,
    ) -> tuple[bool, str]:
        future = client.call_async(Trigger.Request())
        while rclpy.ok() and not future.done():
            if stop_requested_cb() and not ignore_stop_requested:
                break
            rclpy.spin_once(self.node, timeout_sec=0.1)

        if not future.done():
            return False, f'{action_name} request interrupted before completion'

        try:
            response = future.result()
        except Exception as exc:
            return False, f'{action_name} service call failed: {exc}'

        if response is None:
            return False, f'{action_name} service returned no response'

        return bool(response.success), str(response.message)
