#!/usr/bin/env python3
from __future__ import annotations

import math
from collections.abc import Sequence


def _as_float_list(values: Sequence[float], *, name: str, expected_size: int) -> list[float]:
    if len(values) != expected_size:
        raise ValueError(f"{name} must contain {expected_size} values")
    return [float(value) for value in values]


def _as_int_list(values: Sequence[int], *, name: str, expected_size: int) -> list[int]:
    if len(values) != expected_size:
        raise ValueError(f"{name} must contain {expected_size} values")
    return [int(value) for value in values]


def build_observation(
    *,
    cmd_vel: Sequence[float],
    rpy: Sequence[float],
    base_linear_velocity: Sequence[float],
    base_angular_velocity: Sequence[float],
    current_joint_angles: Sequence[float],
    current_joint_velocities: Sequence[float],
    last_action: Sequence[float],
    default_joint_angles: Sequence[float],
    policy_order_indices: Sequence[int],
) -> list[float]:
    cmd_vel = _as_float_list(cmd_vel, name="cmd_vel", expected_size=3)
    rpy = _as_float_list(rpy, name="rpy", expected_size=3)
    roll = rpy[0]
    pitch = rpy[1]
    sin_p = math.sin(pitch)
    cos_p = math.cos(pitch)
    sin_r = math.sin(roll)
    cos_r = math.cos(roll)
    projected_gravity = [
        sin_p,
        -cos_p * sin_r,
        -cos_p * cos_r,
    ]
    base_linear_velocity = _as_float_list(
        base_linear_velocity, name="base_linear_velocity", expected_size=3
    )
    base_angular_velocity = _as_float_list(
        base_angular_velocity, name="base_angular_velocity", expected_size=3
    )
    current_joint_angles = _as_float_list(
        current_joint_angles, name="current_joint_angles", expected_size=12
    )
    current_joint_velocities = _as_float_list(
        current_joint_velocities, name="current_joint_velocities", expected_size=12
    )
    last_action = _as_float_list(last_action, name="last_action", expected_size=12)
    default_joint_angles = _as_float_list(
        default_joint_angles, name="default_joint_angles", expected_size=12
    )
    policy_order_indices = _as_int_list(
        policy_order_indices, name="policy_order_indices", expected_size=12
    )

    observation = [0.0] * 48
    observation[0:3] = cmd_vel
    observation[3:6] = projected_gravity
    observation[6:9] = base_linear_velocity
    observation[9:12] = base_angular_velocity
    for policy_idx, base_idx in enumerate(policy_order_indices):
        if base_idx < 0 or base_idx >= 12:
            raise ValueError("policy_order_indices must contain indices in [0, 11]")
        observation[12 + policy_idx] = (
            current_joint_angles[base_idx] - default_joint_angles[base_idx]
        )
        observation[24 + policy_idx] = current_joint_velocities[base_idx]
        observation[36 + policy_idx] = last_action[policy_idx]

    return observation
