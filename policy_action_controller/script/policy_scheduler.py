#!/usr/bin/env python3
from __future__ import annotations

import sys
import time

import rclpy
from backend_factory import create_control_backend, create_state_adapter
from control_config import load_control_config
from observation_builder import build_observation
from policy_runner import PolicyRunner
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from scheduler_runtime import run_policy_scheduler


class PolicyScheduler(Node):
    def __init__(self) -> None:
        super().__init__('policy_scheduler')
        self.control_cfg = load_control_config()
        self.mode = str(self.control_cfg['mode'])
        self.is_sim2real = self.mode == 'sim2real'
        self.is_sim2sim = self.mode == 'sim2sim'
        self._stop_requested = False
        self._policy_active = False
        self._last_wait_log_time = 0.0

        self.state_adapter = create_state_adapter(self, self.control_cfg)
        self.control_backend = create_control_backend(self, self.control_cfg)
        self.policy_period = 1.0 / self.control_cfg['policy']['cmd_hz']
        self.policy_timer = self.create_timer(self.policy_period, self._on_policy_timer)
        self.policy_runner = self._create_policy_runner()
        self.get_logger().info(f'Policy scheduler started in {self.mode} mode')

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def request_stop(self) -> None:
        self._stop_requested = True

    def start_policy_control(self) -> None:
        self.policy_runner.reset_state()
        self.control_backend.set_gains(
            kp=self.control_cfg['motor']['kp'],
            kd=self.control_cfg['motor']['kd'],
        )
        self.control_backend.enable()
        self._policy_active = True

    def stop_policy_control(self) -> None:
        self._policy_active = False
        self.control_backend.disable()

    def _create_policy_runner(self) -> PolicyRunner:
        model_path = str(self.declare_parameter('model_path', '').value).strip()
        action_scale = float(self.declare_parameter('action_scale', 0.25).value)
        use_cuda = bool(self.declare_parameter('use_cuda', True).value)

        policy_runner = PolicyRunner(
            default_joint_angles=self.control_cfg['default_joint_angles'],
            hard_lower_limits=self.control_cfg['hard_lower_limits'],
            hard_upper_limits=self.control_cfg['hard_upper_limits'],
            policy_order_indices=self.control_cfg['policy_order_indices'],
            model_path=model_path or None,
            action_scale=action_scale,
            use_cuda=use_cuda,
        )
        self.get_logger().info(
            f'Loaded policy model: {policy_runner.model_path} on {policy_runner.device_type}; '
            f'joint_action_scale={policy_runner.action_scale:.6f}'
        )
        return policy_runner

    def _on_policy_timer(self) -> None:
        if not self._policy_active or self.stop_requested:
            return

        state = self.state_adapter.get_policy_state_snapshot()
        if state is None:
            now = time.monotonic()
            if now - self._last_wait_log_time >= 1.0:
                self.get_logger().info(self.state_adapter.wait_message)
                self._last_wait_log_time = now
            return

        try:
            observation = build_observation(
                cmd_vel=state['cmd_vel'],
                rpy=state['rpy'],
                base_linear_velocity=state['base_linear_velocity'],
                base_angular_velocity=state['base_angular_velocity'],
                current_joint_angles=state['current_joint_angles'],
                current_joint_velocities=state['current_joint_velocities'],
                last_action=self.policy_runner.get_last_action(),
                default_joint_angles=self.control_cfg['default_joint_angles'],
                policy_order_indices=self.control_cfg['policy_order_indices'],
            )
            target_joint_angles, _ = self.policy_runner.infer_target_joint_angles(
                observation
            )
        except Exception as exc:
            self.get_logger().error(f'Policy step failed: {exc}')
            self.request_stop()
            self.stop_policy_control()
            return

        self.control_backend.set_target_joint_angles(target_joint_angles)


def main() -> int:
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = PolicyScheduler()
    return run_policy_scheduler(node)


if __name__ == '__main__':
    sys.exit(main())
