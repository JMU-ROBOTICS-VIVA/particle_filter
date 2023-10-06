"""This node broadcasts a tf transform from the the map to odom frame.

See REP 105 for a description of the standard coordinate frame
organization:

https://www.ros.org/reps/rep-0105.html#relationship-between-frames

This is basically a Python port of:

https://github.com/ros-planning/navigation2/blob/73072ae19bfa76c505685a4735ea278d3b499cbe/nav2_amcl/src/amcl_node.cpp#L976

    Subscribes to:
          amcl_pose      (geometry_msgs/PoseWithCovarianceStamped)

    Broadcasts:
          map->frame transform

Author: Nathan Sprague
Version: 10/4/2023
"""
import sys
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseStamped
from geometry_msgs.msg import TransformStamped
import rclpy
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from particle_filter import pf_utils
from tf2_ros import TransformBroadcaster
import tf2_geometry_msgs  # Needs to be here, even though not used.


class AmclTF(rclpy.node.Node):
    def __init__(self):
        super().__init__("amcl_tf")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.create_subscription(
            PoseWithCovarianceStamped, "amcl_pose", self.pose_callback, 10
        )
        self.tf_broadcaster = TransformBroadcaster(self)
        self.get_logger().info("Ready!")
        self.map_to_odom_transform = None
        self.timer = self.create_timer(0.5, self.timer_callback)
        self.rate = 1 / 60

    def timer_callback(self):
        if self.map_to_odom_transform is None:
            self.get_logger().info("Waiting for amcl_pose.")
            return
        self.timer.timer_period_ns = int(self.rate * 1000000000)
        self.map_to_odom_transform.header.stamp = self.get_clock().now().to_msg()
        self.tf_broadcaster.sendTransform(self.map_to_odom_transform)

    def pose_callback(self, pose: PoseWithCovarianceStamped):
        transform = pf_utils.pose_to_transform(pose.pose.pose)
        transform_inv = pf_utils.transform_inverse(transform)
        pose_inv = PoseStamped()
        pose_inv.header = pose.header
        pose_inv.header.frame_id = "base_footprint"
        pose_inv.pose = pf_utils.transform_to_pose(transform_inv)

        try:
            p2 = self.tf_buffer.transform(pose_inv, "odom")

            self.map_to_odom_transform = TransformStamped()
            self.map_to_odom_transform.transform = pf_utils.transform_inverse(
                pf_utils.pose_to_transform(p2.pose)
            )
            self.map_to_odom_transform.header = pose.header
            self.map_to_odom_transform.header.frame_id = "map"
            self.map_to_odom_transform.child_frame_id = "odom"
        except Exception as e:
            self.get_logger().warn(str(e))


def main(args=None):
    rclpy.init(args=args)
    node = AmclTF()
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
