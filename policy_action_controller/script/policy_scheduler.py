#!/usr/bin/env python3
from __future__ import annotations

import math
import signal
import sys
import time
from pathlib import Path
from threading import Lock

import rclpy
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from observation_builder import build_observation
from policy_runner import PolicyRunner
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import Imu, JointState
from simcmd_controller import SimCmdController
from std_srvs.srv import Trigger


class PolicyScheduler(Node):
    def __init__(self) -> None:
        super().__init__('policy_scheduler')
        self.control_cfg = self._load_control_config()
        self.mode = str(self.control_cfg['mode'])
        self.is_sim2real = self.mode == 'sim2real'
        self.is_sim2sim = self.mode == 'sim2sim'
        self._lowstate_msg_type = None
        self._sportstate_msg_type = None
        self._lowcmd_controller_cls = None
        if self.is_sim2real:
            from lowcmd_controller import LowCmdController
            from unitree_go.msg import LowState, SportModeState

            self._lowcmd_controller_cls = LowCmdController
            self._lowstate_msg_type = LowState
            self._sportstate_msg_type = SportModeState
        self.initialize_client = self.create_client(
            Trigger, '/motion_mode_manager/initialize'
        )
        self.exit_client = self.create_client(Trigger, '/motion_mode_manager/exit')
        self._stop_requested = False
        self._state_lock = Lock()
        self._current_joint_angles: list[float] | None = None
        self._current_joint_velocities: list[float] | None = None
        self._base_linear_velocity = [0.0] * 3
        self._base_angular_velocity = [0.0] * 3
        self._rpy = [0.0] * 3
        self._cmd_vel = [0.0] * 3
        self._heading_target: float | None = None
        self._has_motion_state = False
        self._has_imu = False
        self._has_odom = False
        self._has_cmd_vel = False
        self._policy_active = False
        self._last_wait_log_time = 0.0
        self._last_missing_joint_warning_time = 0.0
        self.heading_control_stiffness = float(
            self.declare_parameter('heading_control_stiffness', 0.5).value
        )
        self.max_command_yaw_rate = float(
            self.declare_parameter('max_command_yaw_rate', 1.0).value
        )

        self._create_state_subscriptions()
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            self.control_cfg['topics']['cmd_vel'],
            self._on_cmd_vel,
            10,
        )

        self.motor_controller = self._create_motor_controller()
        self.policy_period = 1.0 / self.control_cfg['policy']['cmd_hz']
        self.policy_timer = self.create_timer(self.policy_period, self._on_policy_timer)
        self.policy_runner = self._create_policy_runner()
        self.get_logger().info(
            f'Policy scheduler started in {self.mode} mode'
        )

    @property
    def stop_requested(self) -> bool:
        return self._stop_requested

    def request_stop(self) -> None:
        self._stop_requested = True

    def get_current_joint_angles(self) -> list[float] | None:
        with self._state_lock:
            if self._current_joint_angles is None:
                return None
            return list(self._current_joint_angles)

    def start_policy_control(self) -> None:
        self.policy_runner.reset_state()
        with self._state_lock:
            self._heading_target = self._rpy[2] if self._has_motion_state else None
        self.motor_controller.set_gains(
            kp=self.control_cfg['motor']['kp'],
            kd=self.control_cfg['motor']['kd'],
        )
        self.motor_controller.enable()
        self._policy_active = True

    def stop_policy_control(self) -> None:
        self._policy_active = False
        self.motor_controller.disable()
        with self._state_lock:
            self._heading_target = None

    def wait_for_current_joint_angles(self, timeout_sec: float = 2.0) -> list[float] | None:
        deadline = time.monotonic() + timeout_sec
        while rclpy.ok() and not self.stop_requested:
            current_joint_angles = self.get_current_joint_angles()
            if current_joint_angles is not None:
                return current_joint_angles
            if time.monotonic() >= deadline:
                break
            rclpy.spin_once(self, timeout_sec=0.05)
        return self.get_current_joint_angles()

    def wait_for_motion_mode_services(self) -> bool:
        while rclpy.ok() and not self.stop_requested:
            if self.initialize_client.wait_for_service(timeout_sec=1.0):
                break
            self.get_logger().info('Waiting for /motion_mode_manager/initialize ...')

        while rclpy.ok() and not self.stop_requested:
            if self.exit_client.wait_for_service(timeout_sec=1.0):
                return True
            self.get_logger().info('Waiting for /motion_mode_manager/exit ...')

        return False

    def initialize_motion_mode(self) -> tuple[bool, str]:
        return self._call_trigger(self.initialize_client, 'initialize')

    def restore_motion_mode(self) -> tuple[bool, str]:
        return self._call_trigger(self.exit_client, 'exit', ignore_stop_requested=True)

    def _create_state_subscriptions(self) -> None:
        topics_cfg = self.control_cfg['topics']
        if self.is_sim2real:
            self.lowstate_sub = self.create_subscription(
                self._lowstate_msg_type,
                topics_cfg['sim2real']['lowstate'],
                self._on_lowstate,
                10,
            )
            self.sportstate_sub = self.create_subscription(
                self._sportstate_msg_type,
                topics_cfg['sim2real']['sportstate'],
                self._on_sportstate,
                10,
            )
            self._wait_for_state_message = (
                f"Waiting for {topics_cfg['sim2real']['lowstate']}, "
                f"{topics_cfg['sim2real']['sportstate']} and "
                f"{topics_cfg['cmd_vel']} before policy inference"
            )
            return

        self.joint_states_sub = self.create_subscription(
            JointState,
            topics_cfg['sim2sim']['joint_states'],
            self._on_joint_states,
            10,
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            topics_cfg['sim2sim']['odom'],
            self._on_odom,
            10,
        )
        self.imu_sub = self.create_subscription(
            Imu,
            topics_cfg['sim2sim']['imu'],
            self._on_imu,
            10,
        )
        self._wait_for_state_message = (
            f"Waiting for {topics_cfg['sim2sim']['joint_states']}, "
            f"{topics_cfg['sim2sim']['odom']}, "
            f"{topics_cfg['sim2sim']['imu']} and "
            f"{topics_cfg['cmd_vel']} before policy inference"
        )

    def _create_motor_controller(self):
        if self.is_sim2real:
            return self._lowcmd_controller_cls(
                self,
                cmd_hz=self.control_cfg['motor']['cmd_hz'],
                target_joint_angles=self.control_cfg['default_joint_angles'],
                real_order_indices=self.control_cfg['real_order_indices'],
                kp=self.control_cfg['stand']['kp'],
                kd=self.control_cfg['stand']['kd'],
            )

        return SimCmdController(
            self,
            cmd_hz=self.control_cfg['motor']['cmd_hz'],
            target_joint_angles=self.control_cfg['default_joint_angles'],
            sim_order_indices=self.control_cfg['sim_order_indices'],
            joint_names=self.control_cfg['joint_names'],
            kp=self.control_cfg['stand']['kp'],
            kd=self.control_cfg['stand']['kd'],
            joint_command_prefix=self.control_cfg['topics']['sim2sim']['joint_command_prefix'],
        )

    def _call_trigger(
        self,
        client,
        action_name: str,
        ignore_stop_requested: bool = False,
    ) -> tuple[bool, str]:
        future = client.call_async(Trigger.Request())
        while rclpy.ok() and not future.done():
            if self.stop_requested and not ignore_stop_requested:
                break
            rclpy.spin_once(self, timeout_sec=0.1)

        if not future.done():
            return False, f'{action_name} request interrupted before completion'

        try:
            response = future.result()
        except Exception as exc:
            return False, f'{action_name} service call failed: {exc}'

        if response is None:
            return False, f'{action_name} service returned no response'

        return bool(response.success), str(response.message)

    def _load_control_config(self) -> dict[str, object]:
        path = self._resolve_config_path()
        data = yaml.safe_load(path.read_text()) or {}
        motor_cfg = dict(data['motor'])
        stand_cfg = dict(data['stand'])
        policy_cfg = dict(data['policy'])
        topics_cfg = dict(data.get('topics') or {})
        sim2real_topics_cfg = dict(topics_cfg.get('sim2real') or {})
        sim2sim_topics_cfg = dict(topics_cfg.get('sim2sim') or {})

        mode = str(data.get('mode', 'sim2real')).strip().lower()
        joint_names = [str(v) for v in data['joint_names']]
        default_joint_angles = [float(v) for v in data['default_joint_angles']]
        hard_lower_limits = [float(v) for v in data['hard_lower_limits']]
        hard_upper_limits = [float(v) for v in data['hard_upper_limits']]
        sim_order_indices = [int(v) for v in data['sim_order_indices']]
        real_order_indices = [int(v) for v in data['real_order_indices']]
        policy_order_indices = [int(v) for v in data['policy_order_indices']]
        topics = {
            'cmd_vel': str(topics_cfg.get('cmd_vel', '/cmd_vel')).strip() or '/cmd_vel',
            'sim2real': {
                'lowstate': str(sim2real_topics_cfg.get('lowstate', '/lowstate')).strip()
                or '/lowstate',
                'sportstate': str(
                    sim2real_topics_cfg.get('sportstate', '/sportmodestate')
                ).strip()
                or '/sportmodestate',
            },
            'sim2sim': {
                'joint_states': str(
                    sim2sim_topics_cfg.get('joint_states', '/joint_states')
                ).strip()
                or '/joint_states',
                'odom': str(sim2sim_topics_cfg.get('odom', '/odom')).strip()
                or '/odom',
                'imu': str(sim2sim_topics_cfg.get('imu', '/imu')).strip() or '/imu',
                'joint_command_prefix': str(
                    sim2sim_topics_cfg.get('joint_command_prefix', '/go2')
                ).strip()
                or '/go2',
            },
        }
        motor = {
            'cmd_hz': float(motor_cfg['cmd_hz']),
            'kp': float(motor_cfg['kp']),
            'kd': float(motor_cfg['kd']),
        }
        stand = {
            'kp': float(stand_cfg['kp']),
            'kd': float(stand_cfg['kd']),
            'duration_sec': float(stand_cfg['duration_sec']),
        }
        policy = {
            'cmd_hz': float(policy_cfg['cmd_hz']),
        }

        if mode not in {'sim2real', 'sim2sim'}:
            raise ValueError("mode must be either 'sim2real' or 'sim2sim'")
        if len(joint_names) != 12:
            raise ValueError('joint_names must contain 12 values')
        if len(default_joint_angles) != 12:
            raise ValueError('default_joint_angles must contain 12 values')
        if len(hard_lower_limits) != 12:
            raise ValueError('hard_lower_limits must contain 12 values')
        if len(hard_upper_limits) != 12:
            raise ValueError('hard_upper_limits must contain 12 values')
        if len(sim_order_indices) != 12:
            raise ValueError('sim_order_indices must contain 12 indices')
        if len(real_order_indices) != 12:
            raise ValueError('real_order_indices must contain 12 indices')
        if len(policy_order_indices) != 12:
            raise ValueError('policy_order_indices must contain 12 indices')
        if motor['cmd_hz'] <= 0.0:
            raise ValueError('motor.cmd_hz must be positive')
        if stand['duration_sec'] <= 0.0:
            raise ValueError('stand.duration_sec must be positive')
        if policy['cmd_hz'] <= 0.0:
            raise ValueError('policy.cmd_hz must be positive')

        return {
            'mode': mode,
            'joint_names': joint_names,
            'default_joint_angles': default_joint_angles,
            'hard_lower_limits': hard_lower_limits,
            'hard_upper_limits': hard_upper_limits,
            'sim_order_indices': sim_order_indices,
            'real_order_indices': real_order_indices,
            'policy_order_indices': policy_order_indices,
            'topics': topics,
            'motor': motor,
            'stand': stand,
            'policy': policy,
        }

    def _resolve_config_path(self) -> Path:
        try:
            return (
                Path(get_package_share_directory('policy_action_controller'))
                / 'config'
                / 'joint_layout.yaml'
            )
        except PackageNotFoundError:
            return Path(__file__).resolve().parents[1] / 'config' / 'joint_layout.yaml'

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
            if self._heading_target is None:
                self._heading_target = self._rpy[2]
            self._has_motion_state = True

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
                self.get_logger().warning(
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
            self._base_angular_velocity = [
                float(msg.angular_velocity.x),
                float(msg.angular_velocity.y),
                float(msg.angular_velocity.z),
            ]
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
            f'joint_action_scale={policy_runner.action_scale:.6f}, '
            f'heading_kp={self.heading_control_stiffness:.3f}'
        )
        return policy_runner

    @staticmethod
    def _wrap_to_pi(angle: float) -> float:
        return math.remainder(angle, 2.0 * math.pi)

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

    def _build_policy_cmd_vel(self, raw_cmd_vel: list[float], yaw: float) -> list[float]:
        with self._state_lock:
            if self._heading_target is None:
                self._heading_target = float(yaw)
            self._heading_target = self._wrap_to_pi(
                self._heading_target + float(raw_cmd_vel[2]) * self.policy_period
            )
            heading_error = self._wrap_to_pi(self._heading_target - float(yaw))

        yaw_rate_cmd = max(
            -self.max_command_yaw_rate,
            min(self.max_command_yaw_rate, self.heading_control_stiffness * heading_error),
        )
        return [float(raw_cmd_vel[0]), float(raw_cmd_vel[1]), float(yaw_rate_cmd)]

    def _get_policy_state_snapshot(self) -> dict[str, list[float]] | None:
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

    def _on_policy_timer(self) -> None:
        if not self._policy_active or self.stop_requested:
            return

        state = self._get_policy_state_snapshot()
        if state is None:
            now = time.monotonic()
            if now - self._last_wait_log_time >= 1.0:
                self.get_logger().info(self._wait_for_state_message)
                self._last_wait_log_time = now
            return

        try:
            policy_cmd_vel = self._build_policy_cmd_vel(state['cmd_vel'], state['rpy'][2])
            observation = build_observation(
                cmd_vel=policy_cmd_vel,
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

        self.motor_controller.set_target_joint_angles(target_joint_angles)


def main() -> int:
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = PolicyScheduler()
    restore_on_shutdown = False

    def handle_sigint(_signum, _frame) -> None:
        if not node.stop_requested:
            if node.is_sim2real:
                node.get_logger().info('Ctrl+C received, restoring mcf before shutdown')
            else:
                node.get_logger().info('Ctrl+C received, stopping policy scheduler')
        node.request_stop()

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        initialize_message = 'sim2sim mode: skipped motion mode initialize'
        if node.is_sim2real:
            if not node.wait_for_motion_mode_services():
                node.get_logger().error('Motion mode services were not available')
                return 1

            success, initialize_message = node.initialize_motion_mode()
            if not success:
                node.get_logger().error(
                    f'Motion mode initialize failed: {initialize_message}'
                )
                return 1

        current_joint_angles = node.wait_for_current_joint_angles(timeout_sec=2.0)
        if current_joint_angles is None:
            if node.is_sim2real:
                joint_source_topic = node.control_cfg['topics']['sim2real']['lowstate']
            else:
                joint_source_topic = node.control_cfg['topics']['sim2sim']['joint_states']
            node.get_logger().error(
                f'Failed to read current joint angles from {joint_source_topic}'
            )
            return 1

        success = node.motor_controller.execute_joint_transition(
            current_joint_angles=current_joint_angles,
            target_joint_angles=node.control_cfg['default_joint_angles'],
            duration_sec=node.control_cfg['stand']['duration_sec'],
        )
        if not success:
            node.get_logger().error('Failed to execute stand transition')
            return 1

        restore_on_shutdown = node.is_sim2real
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
            success, message = node.restore_motion_mode()
            if success:
                node.get_logger().info(f'Motion mode restored: {message}')
            else:
                node.get_logger().error(f'Motion mode restore failed: {message}')

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
