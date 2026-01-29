"""Policy action controller — 将 /joint_target 转发为 per-joint MotorCmd.

启动后先以默认站立姿态初始化关节（持续 INIT_DURATION 秒），
然后切换到订阅 /joint_target 并转发到 /go2/{joint}/command.
始终以固定频率发送关节命令，保持硬件连续反馈。
"""

from __future__ import annotations

import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64MultiArray
from unitree_legged_msgs.msg import MotorCmd

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
JOINTS_ROS2 = [
    'FL_hip_joint',   'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint',   'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint',   'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint',   'RR_thigh_joint', 'RR_calf_joint',
]

# 默认站立位置 — ROS2 顺序（与 obs_preprocess 中 DEFAULT_JOINT_POS 一致）
DEFAULT_POS_ROS2 = [
     0.1,   # FL_hip
     0.8,   # FL_thigh
    -1.5,   # FL_calf
    -0.1,   # FR_hip
     0.8,   # FR_thigh
    -1.5,   # FR_calf
     0.1,   # RL_hip
     1.0,   # RL_thigh
    -1.5,   # RL_calf
    -0.1,   # RR_hip
     1.0,   # RR_thigh
    -1.5,   # RR_calf
]

PMSM = 0x0A          # 位置控制模式
CMD_HZ = 50.0        # 命令发送频率
KP = 50.0            # 位置增益
KD = 2.0             # 速度增益
INIT_DURATION = 5.0  # 初始化阶段持续时间（秒）


class PolicyActionController(Node):
    """初始化关节姿态后，从 /joint_target 读取并转发为 MotorCmd."""

    def __init__(self):
        super().__init__('policy_action_controller')

        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        qos_rt = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        # 当前目标位置（初始为默认站立姿态）
        self._target_pos: list[float] = list(DEFAULT_POS_ROS2)
        # policy 缓存（首次收到消息前为 None）
        self._policy_target: list[float] | None = None

        # 计时器控制 init → policy 切换
        self._init_elapsed = 0.0
        self._init_done = False

        # Per-joint 命令发布器
        self._cmd_pubs: dict[str, rclpy.publisher.Publisher] = {}
        for j in JOINTS_ROS2:
            self._cmd_pubs[j] = self.create_publisher(
                MotorCmd, f'/go2/{j}/command', qos_cmd)

        # 订阅 policy 输出的目标位置
        self.create_subscription(
            Float64MultiArray, '/joint_target',
            self._on_joint_target, qos_rt)

        # --- 监控计数 ---
        self._cmd_count = 0
        self._target_recv_count = 0
        self._last_log_time = time.perf_counter()

        # 固定频率发送命令
        self._timer = self.create_timer(1.0 / CMD_HZ, self._publish_cmds)

        self.get_logger().info(
            f'policy_action_controller 已启动, '
            f'初始化阶段 {INIT_DURATION}s, 发送频率 {CMD_HZ}Hz')

    # ------------------------------------------------------------------
    # 回调
    # ------------------------------------------------------------------

    def _on_joint_target(self, msg: Float64MultiArray):
        """缓存 policy 发来的目标位置（ROS2 顺序, 12 个元素）."""
        if len(msg.data) != 12:
            self.get_logger().warning(
                f'/joint_target 收到 {len(msg.data)} 个值, 期望 12, 忽略.')
            return
        if self._policy_target is None:
            self.get_logger().info('/joint_target 首次收到数据.')
        self._policy_target = list(msg.data)
        self._target_recv_count += 1

    def _publish_cmds(self):
        """定时发送关节命令."""
        dt = 1.0 / CMD_HZ

        # --- 阶段判断 ---
        if not self._init_done:
            # 初始化阶段：固定发送默认站立姿态
            self._init_elapsed += dt
            if self._init_elapsed >= INIT_DURATION:
                self._init_done = True
                self.get_logger().info(
                    f'初始化完成 ({INIT_DURATION}s), 切换到 policy 控制.')
            self._target_pos = list(DEFAULT_POS_ROS2)
        else:
            # Policy 阶段：有数据时切换，无数据时保持上一帧
            if self._policy_target is not None:
                self._target_pos = self._policy_target

        # 逐关节发布 MotorCmd
        for i, j in enumerate(JOINTS_ROS2):
            msg = MotorCmd()
            msg.mode = PMSM
            msg.q = self._target_pos[i]
            msg.dq = 0.0
            msg.tau = 0.0
            msg.kp = KP
            msg.kd = KD
            self._cmd_pubs[j].publish(msg)

        # --- 监控统计 ---
        self._cmd_count += 1
        now = time.perf_counter()
        elapsed = now - self._last_log_time
        if elapsed >= 1.0:
            cmd_hz = self._cmd_count / elapsed
            tgt_hz = self._target_recv_count / elapsed
            phase = 'init' if not self._init_done else 'policy'
            self.get_logger().info(
                f'[{phase}] cmd_hz={cmd_hz:.1f}  '
                f'target_recv_hz={tgt_hz:.1f}  '
                f'(cmd_n={self._cmd_count}, tgt_n={self._target_recv_count})')
            self._cmd_count = 0
            self._target_recv_count = 0
            self._last_log_time = now


def main(args=None):
    rclpy.init(args=args)
    node = PolicyActionController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
