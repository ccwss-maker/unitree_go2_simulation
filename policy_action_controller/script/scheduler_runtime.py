#!/usr/bin/env python3
from __future__ import annotations

import signal

import rclpy


def run_policy_scheduler(node) -> int:
    restore_on_shutdown = False

    def handle_sigint(_signum, _frame) -> None:
        if not node.stop_requested:
            if node.is_sim2real:
                node.get_logger().info('Ctrl+C received, restoring mcf before shutdown')
            else:
                node.get_logger().info('Ctrl+C received, restoring standing pose before shutdown')
        node.request_stop()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        if not node.control_backend.wait_until_ready(lambda: node.stop_requested):
            node.get_logger().error('Required control services were not available')
            return 1

        success, initialize_message = node.control_backend.enter_control_mode(
            lambda: node.stop_requested
        )
        if not success:
            node.get_logger().error(f'Control mode initialize failed: {initialize_message}')
            return 1

        if node.is_sim2real:
            restore_on_shutdown = True

        current_joint_angles = node.state_adapter.wait_for_initial_joint_angles(
            timeout_sec=2.0,
            stop_requested_cb=lambda: node.stop_requested,
        )
        if current_joint_angles is None:
            node.get_logger().error(
                f'Failed to read current joint angles from {node.state_adapter.joint_source_topic}'
            )
            return 1

        success = node.control_backend.execute_joint_transition(
            current_joint_angles=current_joint_angles,
            target_joint_angles=node.control_cfg['default_joint_angles'],
            duration_sec=node.control_cfg['stand']['duration_sec'],
        )
        if not success:
            node.get_logger().error('Failed to execute stand transition')
            return 1

        restore_on_shutdown = True
        node.start_policy_control()
        node.get_logger().info(initialize_message)
        node.get_logger().info(
            f"Policy scheduler running in {node.mode} mode with stand PD: "
            f"kp={node.control_cfg['stand']['kp']}, "
            f"kd={node.control_cfg['stand']['kd']}, "
            f"motor_hz={node.control_cfg['motor']['cmd_hz']}, "
            f"motor PD: kp={node.control_cfg['motor']['kp']}, "
            f"kd={node.control_cfg['motor']['kd']}, "
            f"policy_hz={node.control_cfg['policy']['cmd_hz']}"
        )

        while rclpy.ok() and not node.stop_requested:
            rclpy.spin_once(node, timeout_sec=0.1)

        return 0
    finally:
        if restore_on_shutdown and rclpy.ok():
            node.stop_policy_control()
            current_joint_angles = node.state_adapter.wait_for_initial_joint_angles(
                timeout_sec=0.5,
                stop_requested_cb=lambda: False,
            )
            success, message = node.control_backend.restore_default_mode(
                current_joint_angles=current_joint_angles,
                target_joint_angles=node.control_cfg['default_joint_angles'],
                duration_sec=node.control_cfg['stand']['duration_sec'],
                stop_requested_cb=lambda: False,
                ignore_stop_requested=True,
            )
            if success:
                node.get_logger().info(f'Control mode restored: {message}')
            else:
                node.get_logger().error(f'Control mode restore failed: {message}')

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
