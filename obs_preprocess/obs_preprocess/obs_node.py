"""ROS2 节点 — 订阅 /odom /imu /joint_states /cmd_vel，
整合 48-D 观测 → policy 推理 → 发布关节目标位置.

订阅话题
--------
/odom          nav_msgs/Odometry      → base_lin_vel, base_ang_vel
/imu           sensor_msgs/Imu        → orientation → projected_gravity
/joint_states  sensor_msgs/JointState → joint_pos_rel, joint_vel (12D)
/cmd_vel       geometry_msgs/Twist    → velocity_commands (可选, 默认零)

发布话题
--------
/policy_obs    std_msgs/Float64MultiArray  (48 elements, 监控用)
/joint_target  std_msgs/Float64MultiArray  (12 elements, ROS2 顺序目标位置)
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, JointState
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray

from obs_preprocess.obs_builder import ObsBuilder

# ---------------------------------------------------------------------------
# 关节名 — ROS2 顺序（与 unitree_legged_control.py 中 JOINT_NAMES 一致）
# ---------------------------------------------------------------------------
JOINT_NAMES_ROS2 = [
    'FL_hip_joint',   'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint',   'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint',   'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint',   'RR_thigh_joint', 'RR_calf_joint',
]

# IsaacLab projected_gravity 为单位向量（源码做了 normalize）
GRAVITY_WORLD = np.array([0.0, 0.0, -1.0], dtype=np.float64)


def _gravity_in_body(quat_xyzw: np.ndarray) -> np.ndarray:
    """将世界坐标系重力矢量投影到机器人机身坐标系.

    Parameters
    ----------
    quat_xyzw : (4,)
        机身→世界 方向四元数，ROS 约定 [x, y, z, w].

    Returns
    -------
    gravity_body : (3,)
        重力在机身坐标系的投影.
    """
    x, y, z, w = quat_xyzw
    R = np.array([
        [1 - 2*(y*y + z*z),  2*(x*y - z*w),      2*(x*z + y*w)],
        [2*(x*y + z*w),      1 - 2*(x*x + z*z),  2*(y*z - x*w)],
        [2*(x*z - y*w),      2*(y*z + x*w),      1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R.T @ GRAVITY_WORLD


# ---------------------------------------------------------------------------
# Policy 网络定义 (Actor MLP)
# 架构从 model_2249.pt state_dict 反推：
#   actor.0 Linear(48,128) → ELU → actor.2 Linear(128,128) → ELU
#   → actor.4 Linear(128,128) → ELU → actor.6 Linear(128,12) → tanh
# ---------------------------------------------------------------------------
class ActorMLP(nn.Module):
    """4 层 MLP actor, tanh 输出压缩到 [-1, 1]."""

    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(48, 128),   # actor.0
            nn.ELU(),             # actor.1
            nn.Linear(128, 128),  # actor.2
            nn.ELU(),             # actor.3
            nn.Linear(128, 128),  # actor.4
            nn.ELU(),             # actor.5
            nn.Linear(128, 12),   # actor.6
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs).clamp(-1.0, 1.0)


class ObsNode(Node):
    """订阅传感器话题，组装观测 → policy 推理 → 发布目标关节位置."""

    def __init__(self, model_path: str):
        super().__init__('obs_preprocess_node')

        # --- 加载 policy 模型 ---
        self._device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self._policy = ActorMLP().to(self._device)
        ckpt = torch.load(model_path, map_location=self._device,
                          weights_only=False)
        self._policy.load_state_dict(
            ckpt['model_state_dict'], strict=False)
        self._policy.eval()
        self.get_logger().info(
            f'Policy loaded: {model_path} on {self._device}')

        # --- ObsBuilder ---
        self._builder = ObsBuilder()

        qos_rt = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # 最新传感器快照（首次收到消息前为 None）
        self._base_lin_vel: np.ndarray | None = None
        self._base_ang_vel: np.ndarray | None = None
        self._gravity_body: np.ndarray | None = None
        self._joint_pos_ros2: np.ndarray | None = None
        self._joint_vel_ros2: np.ndarray | None = None
        self._cmd_vel: np.ndarray = np.zeros(3, dtype=np.float64)

        # /joint_states 中 joint_name → 数组下标映射（首条消息时建立）
        self._joint_index: dict[str, int] = {}

        # 订阅
        self.create_subscription(
            Odometry, '/odom', self._on_odom, qos_rt)
        self.create_subscription(
            Imu, '/imu', self._on_imu, qos_rt)
        self.create_subscription(
            JointState, '/joint_states', self._on_joint_states, qos_rt)
        self.create_subscription(
            Twist, '/cmd_vel', self._on_cmd_vel, qos_rt)

        # 发布
        self._obs_pub = self.create_publisher(
            Float64MultiArray, '/policy_obs', qos_rt)
        self._target_pub = self.create_publisher(
            Float64MultiArray, '/joint_target', qos_rt)

        # --- 监控计时 ---
        self._step_count = 0
        self._infer_time_sum = 0.0   # 累计推理耗时 (s)
        self._step_time_sum = 0.0    # 累计 step 总耗时 (s)
        self._last_log_time = time.perf_counter()

        self.get_logger().info('obs_preprocess_node 已启动.')

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------

    def _on_odom(self, msg: Odometry):
        t = msg.twist.twist
        self._base_lin_vel = np.array(
            [t.linear.x, t.linear.y, t.linear.z], dtype=np.float64)
        self._base_ang_vel = np.array(
            [t.angular.x, t.angular.y, t.angular.z], dtype=np.float64)
        self._step()

    def _on_imu(self, msg: Imu):
        q = msg.orientation
        self._gravity_body = _gravity_in_body(
            np.array([q.x, q.y, q.z, q.w], dtype=np.float64))
        self._step()

    def _on_joint_states(self, msg: JointState):
        # 首条消息建立名称→索引映射
        if not self._joint_index:
            for i, name in enumerate(msg.name):
                if name in JOINT_NAMES_ROS2:
                    self._joint_index[name] = i
            if len(self._joint_index) != 12:
                self.get_logger().warning(
                    f'仅映射到 {len(self._joint_index)}/12 个关节, '
                    f'话题包含: {list(msg.name)}')
                return
            self.get_logger().info(
                '已从 /joint_states 映射全部 12 个关节.')

        self._joint_pos_ros2 = np.array(
            [msg.position[self._joint_index[n]] for n in JOINT_NAMES_ROS2],
            dtype=np.float64)
        self._joint_vel_ros2 = np.array(
            [msg.velocity[self._joint_index[n]] for n in JOINT_NAMES_ROS2],
            dtype=np.float64)
        self._step()

    def _on_cmd_vel(self, msg: Twist):
        # cmd_vel = (cmd_vx, cmd_vy, cmd_yaw)
        self._cmd_vel = np.array(
            [msg.linear.x, msg.linear.y, msg.angular.z], dtype=np.float64)

    # ------------------------------------------------------------------
    # 观测组装 → 推理 → 发布
    # ------------------------------------------------------------------

    def _step(self):
        """全部传感器就绪后：build obs → policy 推理 → decode → 发布."""
        if any(v is None for v in (
            self._base_lin_vel,
            self._base_ang_vel,
            self._gravity_body,
            self._joint_pos_ros2,
            self._joint_vel_ros2,
        )):
            return

        t_step_start = time.perf_counter()

        # 1. 组装 48D 观测（包含上一步缓存的 last_action）
        obs = self._builder.build_obs(
            self._joint_pos_ros2,
            self._joint_vel_ros2,
            self._base_lin_vel,
            self._base_ang_vel,
            self._gravity_body,
            self._cmd_vel,
        )

        # 发布 obs（监控用）
        obs_msg = Float64MultiArray()
        obs_msg.data = obs.tolist()
        self._obs_pub.publish(obs_msg)

        # 2. Policy 推理
        t_infer_start = time.perf_counter()
        with torch.no_grad():
            obs_t = torch.from_numpy(obs).float()
            obs_t = obs_t.unsqueeze(0).to(self._device)  # (1, 48)
            action_t = self._policy(obs_t)                # (1, 12)
            action_isaac = action_t.squeeze(0).cpu().numpy()  # (12,)
        t_infer_end = time.perf_counter()

        # 3. decode action → ROS2 顺序目标位置
        #    同时将 action_isaac 缓存为 last_action，下次 build_obs 自动使用
        target_pos_ros2 = self._builder.decode_action(action_isaac)

        # 4. 发布目标位置
        tgt_msg = Float64MultiArray()
        tgt_msg.data = target_pos_ros2.tolist()
        self._target_pub.publish(tgt_msg)

        # --- 监控统计 ---
        t_step_end = time.perf_counter()
        self._infer_time_sum += t_infer_end - t_infer_start
        self._step_time_sum += t_step_end - t_step_start
        self._step_count += 1

        now = time.perf_counter()
        elapsed = now - self._last_log_time
        if elapsed >= 1.0:
            hz = self._step_count / elapsed
            avg_infer_ms = (self._infer_time_sum / self._step_count) * 1000
            avg_step_ms = (self._step_time_sum / self._step_count) * 1000
            self.get_logger().info(
                f'Hz={hz:.1f}  infer={avg_infer_ms:.2f}ms  '
                f'step={avg_step_ms:.2f}ms  (n={self._step_count})')
            self.get_logger().info(
                f'obs  cmd_vel={obs[9:12]}  '
                f'gravity={obs[6:9]}  '
                f'action={action_isaac}')
            self._step_count = 0
            self._infer_time_sum = 0.0
            self._step_time_sum = 0.0
            self._last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    model_path = os.environ.get(
        'POLICY_MODEL_PATH',
        os.path.expanduser('~/unitree_ws/src/model_2249.pt'))
    node = ObsNode(model_path=model_path)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
