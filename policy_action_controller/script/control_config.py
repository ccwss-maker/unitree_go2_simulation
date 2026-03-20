#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory


def resolve_config_path() -> Path:
    try:
        return (
            Path(get_package_share_directory('policy_action_controller'))
            / 'config'
            / 'joint_layout.yaml'
        )
    except PackageNotFoundError:
        return Path(__file__).resolve().parents[1] / 'config' / 'joint_layout.yaml'


def load_control_config() -> dict[str, object]:
    path = resolve_config_path()
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
            'odom': str(sim2sim_topics_cfg.get('odom', '/odom')).strip() or '/odom',
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
