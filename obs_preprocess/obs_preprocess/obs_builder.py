"""Observation pre-processor for Go2 Isaac Sim → ROS2 deployment.

Converts ROS2 sensor streams into the 48-D flat observation vector
expected by the IsaacLab-trained policy network, and decodes the
policy's 12-D action output back into ROS2-ordered joint targets.

Observation layout (48 elements)
--------------------------------
  [ 0: 3]  base_lin_vel        (vx, vy, vz)              from /odom
  [ 3: 6]  base_ang_vel        (wx, wy, wz)              from /odom
  [ 6: 9]  projected_gravity   gravity in body frame      from /imu
  [ 9:12]  velocity_commands   (cmd_vx, cmd_vy, cmd_yaw)
  [12:24]  joint_pos_rel       joint_pos − default_pos    Isaac order
  [24:36]  joint_vel           joint velocities            Isaac order
  [36:48]  last_action         previous policy output      Isaac order
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Joint-order mappings
# ---------------------------------------------------------------------------
# ROS2 order  : [FL_h, FL_t, FL_c, FR_h, FR_t, FR_c,
#                RL_h, RL_t, RL_c, RR_h, RR_t, RR_c]
#
# IsaacLab order: [FL_h, FR_h, RL_h, RR_h,
#                  FL_t, FR_t, RL_t, RR_t,
#                  FL_c, FR_c, RL_c, RR_c]

# Permutation: index into a ROS2-ordered array to produce IsaacLab order.
#   isaac_arr = ros2_arr[ROS2_TO_ISAAC]
ROS2_TO_ISAAC: np.ndarray = np.array(
    [0, 3, 6, 9, 1, 4, 7, 10, 2, 5, 8, 11], dtype=np.int32
)

# Inverse permutation: index into an IsaacLab-ordered array to produce ROS2 order.
#   ros2_arr = isaac_arr[ISAAC_TO_ROS2]
ISAAC_TO_ROS2: np.ndarray = np.array(
    [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11], dtype=np.int32
)

# ---------------------------------------------------------------------------
# Default standing joint positions (IsaacLab order, radians)
# ---------------------------------------------------------------------------
DEFAULT_JOINT_POS: np.ndarray = np.array([
     0.1,   # FL_hip
    -0.1,   # FR_hip
     0.1,   # RL_hip
    -0.1,   # RR_hip
     0.8,   # FL_thigh
     0.8,   # FR_thigh
     1.0,   # RL_thigh
     1.0,   # RR_thigh
    -1.5,   # FL_calf
    -1.5,   # FR_calf
    -1.5,   # RL_calf
    -1.5,   # RR_calf
], dtype=np.float64)

# Action → joint-position scale factor
ACTION_SCALE: float = 0.25

# Observation vector size
OBS_DIM: int = 48


class ObsBuilder:
    """Assembles the 48-D observation vector and decodes policy actions.

    All internal buffers use the IsaacLab joint ordering.  Inputs arriving
    from ROS2 callbacks (ROS2 joint order) are re-indexed on the fly via
    ``ROS2_TO_ISAAC``; action outputs are mapped back with ``ISAAC_TO_ROS2``.
    """

    def __init__(self) -> None:
        self._last_action: np.ndarray = np.zeros(12, dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_obs(
        self,
        joint_pos_ros2: np.ndarray,
        joint_vel_ros2: np.ndarray,
        base_lin_vel: np.ndarray,
        base_ang_vel: np.ndarray,
        gravity_body: np.ndarray,
        cmd_vel: np.ndarray,
    ) -> np.ndarray:
        """Build the 48-D observation vector (IsaacLab joint order).

        Parameters
        ----------
        joint_pos_ros2 : (12,)
            Joint positions in ROS2 index order.
        joint_vel_ros2 : (12,)
            Joint velocities in ROS2 index order.
        base_lin_vel : (3,)
            Base linear velocity [vx, vy, vz].
        base_ang_vel : (3,)
            Base angular velocity [wx, wy, wz].
        gravity_body : (3,)
            Gravity vector projected into the robot body frame.
        cmd_vel : (3,)
            Commanded velocity [cmd_vx, cmd_vy, cmd_yaw].

        Returns
        -------
        obs : (48,)
            Flat observation vector ready for policy inference.
        """
        # Reorder joints from ROS2 → IsaacLab
        joint_pos_isaac = joint_pos_ros2[ROS2_TO_ISAAC]
        joint_vel_isaac = joint_vel_ros2[ROS2_TO_ISAAC]

        # Relative joint positions (subtract default standing pose)
        joint_pos_rel = joint_pos_isaac - DEFAULT_JOINT_POS

        obs = np.empty(OBS_DIM, dtype=np.float64)
        obs[ 0: 3] = base_lin_vel
        obs[ 3: 6] = base_ang_vel
        obs[ 6: 9] = gravity_body
        obs[ 9:12] = cmd_vel
        obs[12:24] = joint_pos_rel
        obs[24:36] = joint_vel_isaac
        obs[36:48] = self._last_action
        return obs

    def decode_action(self, action_isaac: np.ndarray) -> np.ndarray:
        """Decode a policy action into ROS2-ordered target joint positions.

        Caches *action_isaac* as ``last_action`` so the next call to
        ``build_obs`` automatically includes it at obs[36:48].

        Parameters
        ----------
        action_isaac : (12,)
            Raw policy output in [-1, 1], IsaacLab joint order.

        Returns
        -------
        target_pos_ros2 : (12,)
            Target joint positions in ROS2 index order.
        """
        self._last_action = action_isaac.copy()

        target_pos_isaac = DEFAULT_JOINT_POS + action_isaac * ACTION_SCALE
        return target_pos_isaac[ISAAC_TO_ROS2]

    @property
    def last_action(self) -> np.ndarray:
        """Most recent cached action (IsaacLab order), read-only copy."""
        return self._last_action.copy()
