"""Pure-Python particle filter.

AMCL stands for Adaptive Monte Carlo Localization.

This should serve as a (mostly) drop-in replacement for the default
ROS2 localization node: 

https://github.com/ros-planning/navigation2/tree/main/nav2_amcl


    Subscribes to:
          scan           (sensor_msgs/LaserScan)
          map            (nav_msgs/OccupancyGrid)
          odom           (nav_msgs/Odometry)
          initialpose    (geometry_msgs/PoseWithCovarianceStamped)

    Publishes to:
          amcl_pose      (geometry_msgs/PoseWithCovarianceStamped)
          particlecloud    (visualization_msg/MarkerArray)

    ROS Parameters:
      num_particles (int, default: 50)

      update_min_d (double, default: 0.25)
         Minum translation before an update will occur.

      update_min_a (double, default: 0.2)
         Minimum rotation before an update will occur.


      The remaining parameters govern the behavior of the particles.

      z_hit (double, default: 0.5)
         Mixture weight for the z_hit part of the laser scan model.

      z_rand (double, default: 0.5)
         Mixture weight for the z_rand part of the laser scan model.

      sigma_hit (double, default: 0.2 meters)
         Standard deviation for Gaussian model used in z_hit part of
         the laser scan model.

      alpha1 (double, default: 0.2)
         Expected process noise in odometry’s rotation estimate from rotation.

      alpha2 (double, default: 0.2)
         Expected process noise in odometry’s rotation estimate from
         translation.

      alpha3 (double, default: 0.2)
         Expected process noise in odometry’s translation estimate from
         translation.

      alpha4 (double, default: 0.2)
         Expected process noise in odometry’s translation estimate from
         rotation.

      max_beams (int, default: 60)
         How many evenly-spaced beams in each scan to be used when updating 
         the filter.


Some of the documentation for this Node is borrowed from
http://wiki.ros.org/amcl
http://creativecommons.org/licenses/by/3.0/
and
https://navigation.ros.org/configuration/packages/configuring-amcl.html
https://www.apache.org/licenses/LICENSE-2.0
"""
import sys
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from geometry_msgs.msg import PoseArray, PoseWithCovarianceStamped
from visualization_msgs.msg import MarkerArray
from visualization_msgs.msg import Marker
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, QoSDurabilityPolicy
import tf_transformations
import numpy as np
from particle_filter import particle
from particle_filter import pf_utils


class AMCL(Node):
    def __init__(self):
        super().__init__("amcl")

        self.declare_parameter(name="num_particles", value=50)
        self.declare_parameter(name="update_min_a", value=0.2)
        self.declare_parameter(name="update_min_d", value=0.25)
        self.declare_parameter(name="z_hit", value=0.1)
        self.declare_parameter(name="z_rand", value=0.9)
        self.declare_parameter(name="sigma_hit", value=0.2)
        self.declare_parameter(name="alpha1", value=0.2)
        self.declare_parameter(name="alpha2", value=0.2)
        self.declare_parameter(name="alpha3", value=0.2)
        self.declare_parameter(name="alpha4", value=0.2)
        self.declare_parameter(name="max_beams", value=60)

        self.map = None
        self.odom = None
        self.prev_odom = None
        self.particles = []
        self.scan = None

        # Create publishers

        latching_qos = QoSProfile(
            depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL
        )

        self.likelihood_pub = self.create_publisher(
            OccupancyGrid, "likelihood_field", latching_qos
        )

        self.particle_pub = self.create_publisher(MarkerArray, "particlecloud", 10)

        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, "amcl_pose", 10
        )

        # Create subscribers
        self.create_subscription(
            LaserScan, "/scan", self.scan_callback, qos_profile_sensor_data
        )

        self.create_subscription(OccupancyGrid, "/map", self.map_callback, latching_qos)
        self.create_subscription(Odometry, "/odom", self.odom_callback, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self.initial_pose_callback, 10
        )

        self.create_timer(0.1, self.timer_callback)

    def initial_pose_callback(self, initial_pose):
        """This will be called when a new initial pose is provided through
        RViz.  We re-initialize the particle set based on the provided
        pose.

        """
        if self.map is None:
            self.get_logger().info("waiting for map")
            return

        x = initial_pose.pose.pose.position.x
        y = initial_pose.pose.pose.position.y
        yaw = pf_utils.yaw_from_pose(initial_pose.pose.pose)
        x_var = initial_pose.pose.covariance[0]
        y_var = initial_pose.pose.covariance[7]
        yaw_var = initial_pose.pose.covariance[35]
        self.initialize_particles(x, y, yaw, x_var, y_var, yaw_var)

        # Make a rescaled version of the likelihood field used by the
        # particle sensor model that is suitable for visualization:
        pub_map = pf_utils.Map(self.particles[0].likelihood_field.to_msg())
        pub_map.grid = np.array(
            self.particles[0].likelihood_field.grid * 100.0, dtype=int
        )
        self.likelihood_pub.publish(pub_map.to_msg())

    def initialize_particles(self, init_x, init_y, init_yaw, x_var, y_var, yaw_var):
        """Create an initial set of particles.

        The positions and rotations of each particle will be randomly
        drawn from normal distributions based on the input arguments.
        """
        # Get particle parameter values
        num_particles = (
            self.get_parameter("num_particles").get_parameter_value().integer_value
        )
        z_hit = self.get_parameter("z_hit").get_parameter_value().double_value
        z_rand = self.get_parameter("z_rand").get_parameter_value().double_value
        sigma_hit = self.get_parameter("sigma_hit").get_parameter_value().double_value
        alpha1 = self.get_parameter("alpha1").get_parameter_value().double_value
        alpha2 = self.get_parameter("alpha2").get_parameter_value().double_value
        alpha3 = self.get_parameter("alpha3").get_parameter_value().double_value
        alpha4 = self.get_parameter("alpha4").get_parameter_value().double_value
        max_beams = self.get_parameter("max_beams").get_parameter_value().integer_value

        self.particles = []

        for _ in range(num_particles):
            x = init_x + np.random.randn() * x_var
            y = init_y + np.random.randn() * y_var
            yaw = init_yaw + np.random.randn() * yaw_var
            new_particle = particle.Particle(
                x,
                y,
                yaw,
                1.0 / num_particles,
                z_hit,
                z_rand,
                sigma_hit,
                alpha1,
                alpha2,
                alpha3,
                alpha4,
                max_beams,
                self.get_logger(),
                self.get_clock(),
                self.map,
            )
            self.particles.append(new_particle)

        # Publish the newly created particles.
        marker_array = self.create_marker_array_msg()
        self.particle_pub.publish(marker_array)

    def create_marker_array_msg(self, color=(1.0, 0.0, 0.0, 0.5), ns="particles"):
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, p in enumerate(self.particles):
            marker = Marker()
            marker.header.stamp = stamp
            marker.header.frame_id = "map"
            marker.ns = ns
            marker.id = i
            marker.type = marker.ARROW
            marker.pose = p.pose

            scale = len(self.particles) * p.weight * 0.1
            marker.scale.x = scale
            marker.scale.y = marker.scale.z = scale / 5
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]
            marker.color.a = color[3]

            # forever
            marker.lifetime = rclpy.duration.Duration(
                seconds=0.0, nanoseconds=0.0
            ).to_msg()
            markers.markers.append(marker)
        return markers

    def map_callback(self, map_msg):
        """Store the map message in an instance variable."""
        self.map = map_msg

    def odom_callback(self, odom_msg):
        """Store the most recent odom message."""
        if self.odom is None:
            self.prev_odom = odom_msg
        self.odom = odom_msg

    def scan_callback(self, scan_msg):
        """Store the scan message."""
        self.scan = scan_msg

    def odom_changed(self, odom1, odom2):
        """Return true if the robot has moved enough for an update."""

        angle_min = (
            self.get_parameter("update_min_a").get_parameter_value().double_value
        )
        dist_min = self.get_parameter("update_min_d").get_parameter_value().double_value

        p1 = np.array(
            [
                odom1.pose.pose.position.x,
                odom1.pose.pose.position.y,
                odom1.pose.pose.position.z,
            ]
        )
        p2 = np.array(
            [
                odom2.pose.pose.position.x,
                odom2.pose.pose.position.y,
                odom2.pose.pose.position.z,
            ]
        )
        yaw1 = pf_utils.yaw_from_pose(odom1.pose.pose)
        yaw2 = pf_utils.yaw_from_pose(odom2.pose.pose)

        yaw_diff = np.abs(pf_utils.angle_diff(yaw1, yaw2))

        return np.linalg.norm(p1 - p2) > dist_min or yaw_diff > angle_min


    def timer_callback(self):
        if self.scan is None:
            self.get_logger().info("waiting for first scan")
            return

        if self.odom is None:
            self.get_logger().info("waiting for odom")
            return

        if self.map is None:
            self.get_logger().info("waiting for map")
            return

        odom = self.odom
        scan = self.scan

        if not self.odom_changed(self.prev_odom, odom):
            return

        # YOUR CODE HERE! (OR IN HELPER METHODS CALLED FROM HERE)
        # TODO LIST:
        #
        #   * Run one iteration of the particle filter algorithm
        #
        #   * Somehow combine the "votes" of all of the particles
        #     into a single pose estimate.  Note that the message type will
        #     will be PoseWithCovarianceStamped, but you *don't* need to
        #     worry about setting the covariance information.  The most
        #     straightforward thing to do is to average all of the poses.
        #     Averaging angles is a little tricky.  Check out this
        #     Wikipedia page for advice:
        #     https://en.wikipedia.org/wiki/Mean_of_circular_quantities
        #
        #   * Publish the new pose estimate as well as a PoseArray
        #     containing all of the updated particle poses.

        self.prev_odom = odom


def main(args=None):
    rclpy.init(args=args)
    node = AMCL()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    executor.spin()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main(sys.argv)
