#!/usr/bin/env python3
from __future__ import annotations

import ctypes
import time

import rclpy
from rclpy.node import Node
from unitree_go.msg import LowCmd

LOWLEVEL = 0xFF
MOTOR_MODE = 0x01
POS_STOP_F = 2.146e9
VEL_STOP_F = 16000.0


class _RawMotorCmd(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_uint8),
        ("q", ctypes.c_float),
        ("dq", ctypes.c_float),
        ("tau", ctypes.c_float),
        ("kp", ctypes.c_float),
        ("kd", ctypes.c_float),
        ("reserve", ctypes.c_uint32 * 3),
    ]


class _RawBmsCmd(ctypes.Structure):
    _fields_ = [
        ("off", ctypes.c_uint8),
        ("reserve", ctypes.c_uint8 * 3),
    ]


class _RawLowCmd(ctypes.Structure):
    _fields_ = [
        ("head", ctypes.c_uint8 * 2),
        ("level_flag", ctypes.c_uint8),
        ("frame_reserve", ctypes.c_uint8),
        ("sn", ctypes.c_uint32 * 2),
        ("version", ctypes.c_uint32 * 2),
        ("bandwidth", ctypes.c_uint16),
        ("motor_cmd", _RawMotorCmd * 20),
        ("bms_cmd", _RawBmsCmd),
        ("wireless_remote", ctypes.c_uint8 * 40),
        ("led", ctypes.c_uint8 * 12),
        ("fan", ctypes.c_uint8 * 2),
        ("gpio", ctypes.c_uint8),
        ("reserve", ctypes.c_uint32),
        ("crc", ctypes.c_uint32),
    ]


def _crc32_core(words: list[int]) -> int:
    crc = 0xFFFFFFFF
    polynomial = 0x04C11DB7
    for data in words:
        xbit = 1 << 31
        for _ in range(32):
            if crc & 0x80000000:
                crc = ((crc << 1) & 0xFFFFFFFF) ^ polynomial
            else:
                crc = (crc << 1) & 0xFFFFFFFF
            if data & xbit:
                crc ^= polynomial
            xbit >>= 1
    return crc & 0xFFFFFFFF


def _fill_crc(msg: LowCmd) -> None:
    raw = _RawLowCmd()
    raw.head[0] = int(msg.head[0])
    raw.head[1] = int(msg.head[1])
    raw.level_flag = int(msg.level_flag)
    raw.frame_reserve = int(msg.frame_reserve)
    raw.sn[0] = int(msg.sn[0])
    raw.sn[1] = int(msg.sn[1])
    raw.version[0] = int(msg.version[0])
    raw.version[1] = int(msg.version[1])
    raw.bandwidth = int(msg.bandwidth)

    for i in range(20):
        raw.motor_cmd[i].mode = int(msg.motor_cmd[i].mode)
        raw.motor_cmd[i].q = float(msg.motor_cmd[i].q)
        raw.motor_cmd[i].dq = float(msg.motor_cmd[i].dq)
        raw.motor_cmd[i].tau = float(msg.motor_cmd[i].tau)
        raw.motor_cmd[i].kp = float(msg.motor_cmd[i].kp)
        raw.motor_cmd[i].kd = float(msg.motor_cmd[i].kd)
        for j in range(3):
            raw.motor_cmd[i].reserve[j] = int(msg.motor_cmd[i].reserve[j])

    raw.bms_cmd.off = int(msg.bms_cmd.off)
    for i in range(3):
        raw.bms_cmd.reserve[i] = int(msg.bms_cmd.reserve[i])

    for i in range(40):
        raw.wireless_remote[i] = int(msg.wireless_remote[i])
    for i in range(12):
        raw.led[i] = int(msg.led[i])
    for i in range(2):
        raw.fan[i] = int(msg.fan[i])

    raw.gpio = int(msg.gpio)
    raw.reserve = int(msg.reserve)

    size_without_crc = ctypes.sizeof(_RawLowCmd) - ctypes.sizeof(ctypes.c_uint32)
    data = ctypes.string_at(ctypes.byref(raw), size_without_crc)
    words = [
        int.from_bytes(data[i : i + 4], byteorder="little", signed=False)
        for i in range(0, size_without_crc, 4)
    ]
    msg.crc = _crc32_core(words)


class LowCmdController:
    def __init__(
        self,
        node: Node,
        *,
        cmd_hz: float,
        target_joint_angles: list[float],
        real_order_indices: list[int],
        kp: float,
        kd: float,
    ) -> None:
        if len(target_joint_angles) != 12:
            raise ValueError("target_joint_angles must contain 12 values")
        if len(real_order_indices) != 12:
            raise ValueError("real_order_indices must contain 12 indices")
        if cmd_hz <= 0.0:
            raise ValueError("cmd_hz must be positive")

        self.target_joint_angles = [float(v) for v in target_joint_angles]
        self.real_order_indices = [int(v) for v in real_order_indices]
        self.kp = float(kp)
        self.kd = float(kd)
        self.active = False
        self.cmd_period = 1.0 / cmd_hz

        self.publisher = node.create_publisher(LowCmd, "/lowcmd", 10)
        self.lowcmd_msg = self._create_lowcmd_template()
        self.timer = node.create_timer(self.cmd_period, self._publish)

    def enable(self) -> None:
        self.active = True

    def disable(self) -> None:
        self.active = False

    def set_target_joint_angles(self, target_joint_angles: list[float]) -> None:
        if len(target_joint_angles) != 12:
            raise ValueError("target_joint_angles must contain 12 values")
        self.target_joint_angles = [float(v) for v in target_joint_angles]

    def set_gains(self, kp: float, kd: float) -> None:
        self.kp = float(kp)
        self.kd = float(kd)

    def execute_joint_transition(
        self,
        current_joint_angles: list[float],
        target_joint_angles: list[float],
        duration_sec: float,
    ) -> bool:
        if len(current_joint_angles) != 12:
            return False
        if len(target_joint_angles) != 12:
            return False
        if duration_sec <= 0.0:
            return False

        was_active = self.active
        self.active = False

        start_joint_angles = [float(v) for v in current_joint_angles]
        final_joint_angles = [float(v) for v in target_joint_angles]
        start_time = time.monotonic()

        while rclpy.ok():
            elapsed = time.monotonic() - start_time
            alpha = min(1.0, elapsed / duration_sec)
            interp_joint_angles = [
                start + alpha * (target - start)
                for start, target in zip(start_joint_angles, final_joint_angles)
            ]
            self._publish_target(interp_joint_angles)

            if alpha >= 1.0:
                self.target_joint_angles = final_joint_angles
                self.active = was_active
                return True

            time.sleep(self.cmd_period)

        self.active = was_active
        return False

    def _create_lowcmd_template(self) -> LowCmd:
        msg = LowCmd()
        msg.head[0] = 0xFE
        msg.head[1] = 0xEF
        msg.level_flag = LOWLEVEL
        msg.frame_reserve = 0
        msg.gpio = 0

        for motor in msg.motor_cmd:
            motor.mode = MOTOR_MODE
            motor.q = POS_STOP_F
            motor.dq = VEL_STOP_F
            motor.kp = 0.0
            motor.kd = 0.0
            motor.tau = 0.0
            motor.reserve[0] = 0
            motor.reserve[1] = 0
            motor.reserve[2] = 0

        return msg

    def _publish_target(self, target_joint_angles: list[float]) -> None:
        for motor_idx in range(12):
            joint_idx = self.real_order_indices[motor_idx]
            motor = self.lowcmd_msg.motor_cmd[motor_idx]
            motor.mode = MOTOR_MODE
            motor.q = float(target_joint_angles[joint_idx])
            motor.dq = 0.0
            motor.kp = self.kp
            motor.kd = self.kd
            motor.tau = 0.0

        _fill_crc(self.lowcmd_msg)
        self.publisher.publish(self.lowcmd_msg)

    def _publish(self) -> None:
        if not self.active:
            return
        self._publish_target(self.target_joint_angles)
