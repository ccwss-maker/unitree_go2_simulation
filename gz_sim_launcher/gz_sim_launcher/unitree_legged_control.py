"""
Unitree legged control — ROS2 plugin.

Per-joint PD + feedforward torque controller.
Subscribes to /go2/{joint}/command (MotorCmd), reads /joint_states,
computes torque, publishes to /go2/{joint}/cmd_force and /go2/{joint}/state.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from unitree_legged_msgs.msg import MotorCmd, MotorState

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PMSM = 0x0A
BRAKE = 0x00
POS_STOP_F = 2.146e9   # sentinel: disable position term
VEL_STOP_F = 16000.0   # sentinel: disable velocity term

# Joint limits from model.sdf  (lower, upper, effort, velocity)
JOINT_LIMITS = {
    'hip':   (-1.0472, 1.0472, 23.7, 30.1),
    'thigh': (-1.5708, 3.4907, 23.7, 30.1),
    'calf':  (-2.7227, -0.8378, 35.55, 20.06),
}

JOINT_NAMES = [
    'FL_hip_joint', 'FL_thigh_joint', 'FL_calf_joint',
    'FR_hip_joint', 'FR_thigh_joint', 'FR_calf_joint',
    'RL_hip_joint', 'RL_thigh_joint', 'RL_calf_joint',
    'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
]
# JOINT_NAMES = [
#     'RR_hip_joint', 'RR_thigh_joint', 'RR_calf_joint',
# ]

def _joint_type(joint_name: str) -> str:
    if 'hip' in joint_name:
        return 'hip'
    if 'calf' in joint_name:
        return 'calf'
    return 'thigh'


def _clamp(val, lo, hi):
    return max(lo, min(val, hi))


class JointState_:
    """Per-joint runtime state mirror (matches ROS1 lastState + lastCmd)."""
    __slots__ = ('q', 'dq', 'tau_est', 'last_cmd', 'initialized')

    def __init__(self):
        self.q = 0.0
        self.dq = 0.0
        self.tau_est = 0.0
        self.last_cmd = MotorCmd()
        self.initialized = False


class UnitreeLeggedControl(Node):

    def __init__(self):
        super().__init__('unitree_legged_control')

        # QoS: best-effort for sensor feedback / state publish
        qos_rt = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # QoS for cmd_force -> ros_gz_bridge requires RELIABLE
        qos_force = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # QoS for incoming MotorCmd commands
        qos_cmd = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
        )

        self._states = {}          # joint_name -> JointState_
        self._cmd_pubs = {}        # joint_name -> Publisher[Float64]   (cmd_force)
        self._state_pubs = {}      # joint_name -> Publisher[MotorState]
        self._cmd_subs = {}        # joint_name -> Subscriber[MotorCmd]
        self._limits = {}          # joint_name -> (lower, upper, effort, velocity)

        # Pre-build joint name -> index map for /joint_states lookup
        self._joint_index = {}     # filled on first /joint_states message

        for jname in JOINT_NAMES:
            jtype = _joint_type(jname)
            self._limits[jname] = JOINT_LIMITS[jtype]
            self._states[jname] = JointState_()

            # Publishers
            self._cmd_pubs[jname] = self.create_publisher(
                Float64, f'/go2/{jname}/cmd_force', qos_force)
            self._state_pubs[jname] = self.create_publisher(
                MotorState, f'/go2/{jname}/state', qos_rt)

            # Per-joint command subscriber
            self._cmd_subs[jname] = self.create_subscription(
                MotorCmd, f'/go2/{jname}/command',
                lambda msg, j=jname: self._on_command(j, msg), qos_cmd)

        # Single subscriber for joint state feedback from Gazebo
        self._joint_states_sub = self.create_subscription(
            JointState, '/joint_states', self._on_joint_states, qos_rt)

        self._last_stamp = None  # for dt calculation from stamp

        self.get_logger().info('unitree_legged_control started, waiting for /joint_states...')

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _on_command(self, joint_name: str, msg: MotorCmd):
        """Store incoming MotorCmd (non-RT safe in Python, acceptable here)."""
        self._states[joint_name].last_cmd = msg

    def _on_joint_states(self, msg: JointState):
        """Main control loop — triggered once per /joint_states publish."""

        # Build index map on first call
        if not self._joint_index:
            for i, name in enumerate(msg.name):
                if name in self._states:
                    self._joint_index[name] = i
            # Initialize state.q to current position (mirrors ROS1 starting())
            for jname, idx in self._joint_index.items():
                st = self._states[jname]
                if not st.initialized:
                    st.q = msg.position[idx]
                    st.last_cmd.q = msg.position[idx]
                    st.initialized = True
            self.get_logger().info(
                f'Mapped {len(self._joint_index)} joints from /joint_states.')

        # Compute dt from message stamp
        stamp = msg.header.stamp
        current_time = stamp.sec + stamp.nanosec * 1e-9
        if self._last_stamp is not None:
            dt = current_time - self._last_stamp
            if dt <= 0.0 or dt > 0.1:
                dt = 0.005  # fallback to physics step size
        else:
            dt = 0.005  # first iteration, use physics step size
        self._last_stamp = current_time

        for jname, idx in self._joint_index.items():
            st = self._states[jname]
            cmd = st.last_cmd
            lo, hi, eff_lim, vel_lim = self._limits[jname]

            current_pos = msg.position[idx]

            # --- Compute velocity (same filter as ROS1) ---
            # vel = lastVel*0.35 + 0.65*(currentPos - lastPos) / dt
            current_vel = st.dq * 0.35 + 0.65 * (current_pos - st.q) / dt

            # --- Process command based on mode ---
            if cmd.mode == PMSM:
                target_pos = _clamp(cmd.q, lo, hi)
                pos_stiffness = cmd.kp
                if abs(cmd.q - POS_STOP_F) < 1e-6:
                    pos_stiffness = 0.0

                target_vel = _clamp(cmd.dq, -vel_lim, vel_lim)
                vel_stiffness = cmd.kd
                if abs(cmd.dq - VEL_STOP_F) < 1e-6:
                    vel_stiffness = 0.0

                target_torque = _clamp(cmd.tau, -eff_lim, eff_lim)

            elif cmd.mode == BRAKE:
                target_pos = current_pos
                pos_stiffness = 0.0
                target_vel = 0.0
                vel_stiffness = 0.0
                target_torque = 0.0

            else:
                # Unknown mode or not yet set — hold position with zero gains
                target_pos = current_pos
                pos_stiffness = 0.0
                target_vel = 0.0
                vel_stiffness = 0.0
                target_torque = 0.0

            # --- Compute torque (same law as ROS1 computeTorque) ---
            calc_torque = (pos_stiffness * (target_pos - current_pos)
                           + vel_stiffness * (target_vel - current_vel)
                           + target_torque)
            calc_torque = _clamp(calc_torque, -eff_lim, eff_lim)

            # --- Send torque command ---
            force_msg = Float64()
            force_msg.data = calc_torque
            self._cmd_pubs[jname].publish(force_msg)

            # --- Update internal state ---
            st.q = current_pos
            st.dq = current_vel
            st.tau_est = msg.effort[idx] if idx < len(msg.effort) else calc_torque

            # --- Publish MotorState ---
            state_msg = MotorState()
            state_msg.mode = cmd.mode
            state_msg.q = float(st.q)
            state_msg.dq = float(st.dq)
            state_msg.tauest = float(st.tau_est)
            self._state_pubs[jname].publish(state_msg)


def main(args=None):
    rclpy.init(args=args)
    node = UnitreeLeggedControl()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
