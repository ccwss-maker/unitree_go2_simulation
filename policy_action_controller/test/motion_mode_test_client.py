#!/usr/bin/env python3
from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class MotionModeTestClient(Node):
    def __init__(self) -> None:
        super().__init__("motion_mode_test_client")
        self.initialize_client = self.create_client(Trigger, "/motion_mode_manager/initialize")
        self.exit_client = self.create_client(Trigger, "/motion_mode_manager/exit")

    def wait_for_services(self) -> None:
        while rclpy.ok() and not self.initialize_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /motion_mode_manager/initialize ...")
        while rclpy.ok() and not self.exit_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Waiting for /motion_mode_manager/exit ...")

    def call_initialize(self) -> tuple[bool, str]:
        return self._call(self.initialize_client)

    def call_exit(self) -> tuple[bool, str]:
        return self._call(self.exit_client)

    def _call(self, client) -> tuple[bool, str]:
        future = client.call_async(Trigger.Request())
        while rclpy.ok() and not future.done():
            rclpy.spin_once(self, timeout_sec=0.1)

        if not future.done():
            return False, "Service call did not complete"

        response = future.result()
        if response is None:
            return False, "Service returned no response"

        return bool(response.success), str(response.message)


def main() -> None:
    rclpy.init()
    node = MotionModeTestClient()

    try:
        node.wait_for_services()
        while rclpy.ok():
            cmd = input("Input 1=initialize, 2=exit, q=quit > ").strip().lower()
            if cmd == "1":
                success, message = node.call_initialize()
                print(f"[initialize] success={success} message={message}")
            elif cmd == "2":
                success, message = node.call_exit()
                print(f"[exit] success={success} message={message}")
            elif cmd in {"q", "quit"}:
                break
            else:
                print("Unknown input, use 1 / 2 / q")
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
