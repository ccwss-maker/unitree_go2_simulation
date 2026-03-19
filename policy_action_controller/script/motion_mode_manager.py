#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from threading import Event, Lock

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_srvs.srv import Trigger

from unitree_api.msg import Request, Response


@dataclass
class _PendingCall:
    api_id: int
    event: Event
    response: Response | None = None


class MotionModeManager(Node):
    API_ID_CHECK_MODE = 1001
    API_ID_SELECT_MODE = 1002
    API_ID_RELEASE_MODE = 1003

    def __init__(self) -> None:
        super().__init__("motion_mode_manager")

        self.request_timeout = float(self.declare_parameter("request_timeout_sec", 5.0).value)
        self.restore_timeout = float(self.declare_parameter("restore_timeout_sec", 8.0).value)
        self.restore_mode = str(self.declare_parameter("restore_mode", "mcf").value)

        self.response_group = ReentrantCallbackGroup()
        self.service_group = MutuallyExclusiveCallbackGroup()

        self.request_pub = self.create_publisher(Request, "/api/motion_switcher/request", 10)
        self.response_sub = self.create_subscription(
            Response,
            "/api/motion_switcher/response",
            self._on_response,
            10,
            callback_group=self.response_group,
        )
        self.initialize_srv = self.create_service(
            Trigger,
            "/motion_mode_manager/initialize",
            self._handle_initialize,
            callback_group=self.service_group,
        )
        self.exit_srv = self.create_service(
            Trigger,
            "/motion_mode_manager/exit",
            self._handle_exit,
            callback_group=self.service_group,
        )

        self._pending_calls: dict[int, _PendingCall] = {}
        self._pending_lock = Lock()

    def initialize_motion_mode(self) -> tuple[bool, str]:
        ret, name, form = self.check_mode()
        if ret != 0:
            message = f"CheckMode failed, ret={ret}"
            self.get_logger().warn(message)
            return False, message

        self.get_logger().info(f"Current mode: name='{name}', form='{form}'")
        if name != "mcf":
            message = f"Current mode already cleared: name='{name}', form='{form}'"
            self.get_logger().info(message)
            return True, message

        self.get_logger().info("mcf detected, calling ReleaseMode...")
        ret = self.release_mode()
        if ret != 0:
            message = f"ReleaseMode failed, ret={ret}"
            self.get_logger().warn(message)
            return False, message

        self.get_logger().info("ReleaseMode success")
        return self._wait_until_mode_not("mcf", self.request_timeout)

    def restore_mode_and_wait(self, timeout_sec: float | None = None) -> tuple[bool, str]:
        deadline = time.monotonic() + (timeout_sec if timeout_sec is not None else self.restore_timeout)
        self.get_logger().info(f"Restoring mode '{self.restore_mode}'...")

        while time.monotonic() < deadline and rclpy.ok():
            self.select_mode_noreply(self.restore_mode)
            time.sleep(0.1)

            ret, name, _form = self.check_mode()
            if ret == 0 and name == self.restore_mode:
                message = f"Mode '{self.restore_mode}' restored"
                self.get_logger().info(message)
                return True, message
            if ret != 0:
                self.get_logger().warn(f"CheckMode failed during restore, ret={ret}")

        message = f"Mode '{self.restore_mode}' not detected before timeout"
        self.get_logger().warn(message)
        return False, message

    def check_mode(self) -> tuple[int, str, str]:
        req = self._make_request(self.API_ID_CHECK_MODE)
        response = self._call(req)
        if response is None:
            return -1, "", ""
        if response.header.status.code != 0:
            return int(response.header.status.code), "", ""
        try:
            data = json.loads(response.data) if response.data else {}
        except json.JSONDecodeError:
            return -2, "", ""
        return 0, str(data.get("name", "")), str(data.get("form", ""))

    def release_mode(self) -> int:
        req = self._make_request(self.API_ID_RELEASE_MODE)
        response = self._call(req)
        if response is None:
            return -1
        return int(response.header.status.code)

    def select_mode_noreply(self, name: str) -> None:
        req = self._make_request(self.API_ID_SELECT_MODE, {"name": name}, noreply=True)
        self.request_pub.publish(req)

    def _make_request(
        self,
        api_id: int,
        parameter: dict[str, str] | None = None,
        noreply: bool = False,
    ) -> Request:
        req = Request()
        req.header.identity.api_id = api_id
        req.header.identity.id = time.monotonic_ns()
        req.header.policy.noreply = noreply
        if parameter is not None:
            req.parameter = json.dumps(parameter)
        return req

    def _call(self, req: Request) -> Response | None:
        pending = _PendingCall(api_id=req.header.identity.api_id, event=Event())
        with self._pending_lock:
            self._pending_calls[req.header.identity.id] = pending

        self.request_pub.publish(req)
        pending.event.wait(timeout=self.request_timeout)

        with self._pending_lock:
            stored = self._pending_calls.pop(req.header.identity.id, None)

        if stored is None or stored.response is None:
            return None
        return stored.response

    def _on_response(self, msg: Response) -> None:
        request_id = msg.header.identity.id
        with self._pending_lock:
            pending = self._pending_calls.get(request_id)
        if pending is None:
            return
        if msg.header.identity.api_id != pending.api_id:
            return
        pending.response = msg
        pending.event.set()

    def _wait_until_mode_not(self, mode_name: str, timeout_sec: float) -> tuple[bool, str]:
        deadline = time.monotonic() + timeout_sec
        while time.monotonic() < deadline and rclpy.ok():
            ret, name, form = self.check_mode()
            if ret != 0:
                message = f"CheckMode failed after release, ret={ret}"
                self.get_logger().warn(message)
                return False, message
            if name != mode_name:
                message = f"Mode cleared successfully: name='{name}', form='{form}'"
                self.get_logger().info(message)
                return True, message
            time.sleep(0.1)

        message = f"Mode '{mode_name}' is still active after release timeout"
        self.get_logger().warn(message)
        return False, message

    def _handle_initialize(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        success, message = self.initialize_motion_mode()
        response.success = success
        response.message = message
        return response

    def _handle_exit(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        success, message = self.restore_mode_and_wait()
        response.success = success
        response.message = message
        return response


def main() -> None:
    rclpy.init()
    node = MotionModeManager()
    executor = MultiThreadedExecutor(num_threads=2)
    executor.add_node(node)

    try:
        node.get_logger().info("Services ready: /motion_mode_manager/initialize, /motion_mode_manager/exit")
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
