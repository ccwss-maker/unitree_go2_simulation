#!/usr/bin/env python3
"""
Contact Force Processor Node for Go2 Robot

Subscribes to bridged contact sensor data from Gazebo Sim and processes
it to publish aggregated force measurements, matching the behavior of
the original UnitreeFootContactPlugin.

Original plugin behavior (foot_contact_plugin.cc):
- Aggregates forces from all contact points by summing and averaging
- Uses wrench(0) index for force data
- Publishes to /visual/{sensor_name}/the_force
- Force is in local coordinates (body_1_wrench)
"""

import rclpy
from rclpy.node import Node
from ros_gz_interfaces.msg import Contacts
from geometry_msgs.msg import WrenchStamped


class FootContactProcessor(Node):
    """Processes contact sensor data and publishes averaged forces"""

    def __init__(self):
        super().__init__('foot_contact_processor')

        # Foot names matching the robot model
        self.feet = ['FL', 'FR', 'RL', 'RR']

        # Create subscribers for bridged contact sensors
        self.contact_subs = {}
        for foot in self.feet:
            self.contact_subs[foot] = self.create_subscription(
                Contacts,
                f'/go2/foot_contact/{foot}',
                lambda msg, f=foot: self.contact_callback(msg, f),
                10
            )

        # Create publishers for processed forces
        # Topic format matches original: /visual/{sensor_name}/the_force
        self.force_pubs = {}
        for foot in self.feet:
            self.force_pubs[foot] = self.create_publisher(
                WrenchStamped,
                f'/visual/{foot}_foot_contact/the_force',
                10
            )

        self.get_logger().info('Foot contact processor node started')
        self.get_logger().info(f'Subscribed to contact sensors: {", ".join(self.feet)}')

    def contact_callback(self, msg: Contacts, foot_name: str):
        """
        Process contact data and publish averaged forces

        Matches original plugin behavior:
        - If no contacts: publish zero forces
        - If contacts exist: sum all forces and divide by contact count
        - Always use wrench index 0 for each contact

        Args:
            msg: Contacts message from Gazebo Sim (bridged to ROS)
            foot_name: Name of the foot (FL, FR, RL, RR)
        """
        wrench = WrenchStamped()
        wrench.header.stamp = self.get_clock().now().to_msg()
        wrench.header.frame_id = f'{foot_name}_foot'

        # Count contacts
        contact_count = len(msg.contacts)

        if contact_count == 0:
            # No contact - publish zero forces
            # (wrench.wrench.force already initialized to 0)
            self.force_pubs[foot_name].publish(wrench)
            return

        # Aggregate forces from all contact points
        fx_sum = 0.0
        fy_sum = 0.0
        fz_sum = 0.0

        for contact in msg.contacts:
            # Original plugin checks position_size == 1
            if len(contact.positions) != 1:
                self.get_logger().warn(
                    f'{foot_name}: Contact position count is {len(contact.positions)}, expected 1'
                )

            # Original plugin uses wrench(0) - always index 0
            if len(contact.wrenches) > 0:
                # Use body_1_wrench to match original plugin
                # Force is in local coordinates
                wrench_data = contact.wrenches[0]
                fx_sum += wrench_data.body_1_wrench.force.x
                fy_sum += wrench_data.body_1_wrench.force.y
                fz_sum += wrench_data.body_1_wrench.force.z

        # Average the forces (match original: Fx/count, Fy/count, Fz/count)
        wrench.wrench.force.x = fx_sum / float(contact_count)
        wrench.wrench.force.y = fy_sum / float(contact_count)
        wrench.wrench.force.z = fz_sum / float(contact_count)

        self.force_pubs[foot_name].publish(wrench)


def main(args=None):
    rclpy.init(args=args)
    node = FootContactProcessor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
