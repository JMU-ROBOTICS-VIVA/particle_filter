"""A Particle class.

The class below implements the standard laser sensor model
and differential drive motion model.

Author: Nathan Sprague
Version: 10/4/2023

"""

import copy
import rclpy
import numpy as np
import math
import tf_transformations
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker, MarkerArray
from particle_filter import pf_utils


class Particle:
    """Particles are essentially wrappers around geometry_msgs/Pose
    objects.
    """

    # This is a class-variable because it makes sense for all particles
    # to share the same map.
    likelihood_field = None

    def __init__( self, x, y, yaw, weight, laser_z_hit, laser_z_rand,
                  laser_sigma_hit, alpha1, alpha2, alpha3, alpha4, max_beams,
                  logger, clock, map_msg=None, ):
        """Initialize the particle."""

        self.weight = weight
        self.laser_z_hit = laser_z_hit
        self.laser_z_rand = laser_z_rand
        self.laser_sigma_hit = laser_sigma_hit
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4
        self.max_beams = max_beams
        self.logger = logger
        self.clock = clock
        self._pose = Pose()

        # The next three lines are calling the property setters.
        self.x = x
        self.y = y
        self.yaw = yaw

        if Particle.likelihood_field is None:
            self.update_likelihood_field(map_msg, self.laser_sigma_hit)

    def update_likelihood_field(self, map_msg, laser_sigma):
        """The likelihood field is essentially a map indicating which
        locations are likely to result in a laser hit.  Points that
        are occupied in map_msg have the highest probability, points
        that are farther away from occupied regions have a lower
        probability.  The laser_sigma argument controls how quickly
        the probability falls off.  Ideally, it should be related to
        the precision of the laser scanner being used.

        """

        self.logger.info("building Likelihood map...")
        world_map = pf_utils.Map(map_msg)

        self.logger.info("building KDTree")
        from sklearn.neighbors import KDTree

        occupied_points = []
        all_positions = []
        for i in range(world_map.grid.shape[0]):
            for j in range(world_map.grid.shape[1]):
                all_positions.append(world_map.cell_position(i, j))
                if world_map.grid[i, j] > 90:
                    occupied_points.append(world_map.cell_position(i, j))

        kdt = KDTree(np.array(occupied_points))

        self.logger.info("Constructing likelihood field from KDTree.")
        likelihood_field = pf_utils.Map(world_map.to_msg())
        dists = kdt.query(all_positions, k=1)[0][:]
        probs = np.exp(-(dists**2) / (2 * laser_sigma**2))
        likelihood_field.grid = probs.reshape(likelihood_field.grid.shape)

        self.logger.info("Done building likelihood field")
        Particle.likelihood_field = likelihood_field

    def copy(self):
        """Return a deep copy of this particle.  This needs to be used when
        resampling to ensure that we don't end up with aliased
        particles in the particle set.

        """
        return Particle(
            self.x,
            self.y,
            self.yaw,
            1.0,
            self.laser_z_hit,
            self.laser_z_rand,
            self.laser_sigma_hit,
            self.alpha1,
            self.alpha2,
            self.alpha3,
            self.alpha4,
            self.max_beams,
            self.logger,
            self.clock,
        )

    @property
    def x(self):
        """x position in meters"""
        return self._pose.position.x

    @x.setter
    def x(self, x):
        self._pose.position.x = x

    @property
    def y(self):
        """y position in meters"""
        return self._pose.position.y

    @y.setter
    def y(self, y):
        self._pose.position.y = y

    @property
    def yaw(self):
        """Orientation in radians."""
        return pf_utils.yaw_from_pose(self._pose)

    @yaw.setter
    def yaw(self, yaw):
        quat = tf_transformations.quaternion_from_euler(0, 0, yaw)
        self._pose.orientation.x = quat[0]
        self._pose.orientation.y = quat[1]
        self._pose.orientation.z = quat[2]
        self._pose.orientation.w = quat[3]

    @property
    def pose(self):
        """Pose of this particle as a geometry_msgs/Pose object"""
        return self._pose

    @pose.setter
    def pose(self, pose):
        self._pose = copy.copy(pose)

        # Now normalize the quaternion.
        quat = np.array(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ]
        )
        quat = tf_transformations.unit_vector(quat)
        self._pose.orientation.x = quat[0]
        self._pose.orientation.y = quat[1]
        self._pose.orientation.z = quat[2]
        self._pose.orientation.w = quat[3]

    def sense(self, scan_msg):
        """Update the weight of this particle based on a LaserScan message.
        The new weight will be relatively high if the pose of this
        particle corresponds will with the scan, it will be relatively
        low if the pose does not correspond to this scan.

        The algorithm used here is loosely based on the Algorithm in
        Table 6.3 of Probabilistic Robotics Thrun et. al. 2005

        Arguments:
           scan_msg - sensor_msgs/LaserScan object

        Returns:
           None

        """
        xs, ys = self._scan_to_endpoints(scan_msg)
        total_prob = 0
        for i in range(0, len(xs), math.ceil(len(xs) / self.max_beams)):
            likelihood = self.likelihood_field.get_cell(xs[i], ys[i])
            if np.isnan(likelihood):
                likelihood = 0
            total_prob += np.log(self.laser_z_hit * likelihood + self.laser_z_rand)
        self.weight *= np.exp(total_prob)

    def delta(self, odom1, odom2):
        yaw1 = pf_utils.yaw_from_pose(odom1.pose.pose)
        yaw2 = pf_utils.yaw_from_pose(odom2.pose.pose)
        x_d = odom2.pose.pose.position.x - odom1.pose.pose.position.x
        y_d = odom2.pose.pose.position.y - odom1.pose.pose.position.y
        yaw_d = pf_utils.angle_diff(yaw2, yaw1)
        return (x_d, y_d, yaw_d)

    def move(self, odom1, odom2):
        """Move the particle according to the observed odometry.

        See Probalistic Robotics p. 136 for a full description of the
        algorithm and this link for a C++ implementation:
        https://github.com/ros-planning/navigation2/blob/73072ae19bfa76c505685a4735ea278d3b499cbe/nav2_amcl/src/motion_model/differential_motion_model.cpp#L40
        """

        delta = self.delta(odom1, odom2)
        yaw1 = pf_utils.yaw_from_pose(odom1.pose.pose)

        d_trans = np.sqrt(delta[0] ** 2 + delta[1] ** 2)
        if d_trans < 0.01:
            d_rot1 = 0
        else:
            d_rot1 = pf_utils.angle_diff(np.arctan2(delta[1], delta[0]), yaw1)
        d_rot2 = pf_utils.angle_diff(delta[2], d_rot1)

        d_rot1_hat = d_rot1 - (
            np.random.randn() * (self.alpha1 * d_rot1**2 + self.alpha2 * d_trans**2)
        )

        d_trans_hat = d_trans - (
            np.random.randn()
            * (
                self.alpha3 * d_trans**2
                + self.alpha4 * d_rot1**2
                + self.alpha4 * d_rot2**2
            )
        )

        d_rot2_hat = d_rot2 - (
            np.random.randn() * (self.alpha1 * d_rot2**2 + self.alpha2 * d_trans**2)
        )

        self.x += d_trans_hat * np.cos(self.yaw + d_rot1_hat)
        self.y += d_trans_hat * np.sin(self.yaw + d_rot1_hat)
        self.yaw += d_rot1_hat + d_rot2_hat

    def _scan_to_endpoints(self, scan_msg):
        """Helper method used to convert convert range values into x, y
        coordinates in the map coordinate frame.  Based on
        probabilistic robotics equation 6.32

        """
        yaw_beam = np.arange(
            scan_msg.angle_min, scan_msg.angle_max, scan_msg.angle_increment
        )
        ranges = np.array(scan_msg.ranges)
        xs = self.x + ranges * np.cos(self.yaw + yaw_beam)
        ys = self.y + ranges * np.sin(self.yaw + yaw_beam)

        # Clear out nan entries:
        xs = xs[np.logical_not(np.isnan(xs))]
        ys = ys[np.logical_not(np.isnan(ys))]

        # Clear out inf entries:
        xs = xs[np.logical_not(np.isinf(xs))]
        ys = ys[np.logical_not(np.isinf(ys))]
        return xs, ys

    def scan_markers(self, scan_msg, color=(0, 1.0, 0)):
        """Returns a MarkerArray message displaying what the scan message
        would look like from the perspective of this particle.  Just
        for debugging.

        Returns:
           visualization_msgs/MarkerArray

        """

        xs, ys = self._scan_to_endpoints(scan_msg)
        marker_array = MarkerArray()
        header = Marker().header
        header.stamp = self.clock.now().to_msg()
        header.frame_id = "map"
        for i in range(len(xs)):
            marker = Marker()
            marker.header = header
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            marker.pose.position.x = xs[i]
            marker.pose.position.y = ys[i]
            marker.pose.position.z = 0.3
            marker.pose.orientation.w = 1.0
            marker.id = np.array([(id(self) * 13 * i * 17) % 2**32], dtype="int32")[0]
            marker.scale.x = 0.02
            marker.scale.y = 0.02
            marker.scale.z = 0.02

            marker.color.a = 1.0
            marker.color.r = color[0]
            marker.color.g = color[1]
            marker.color.b = color[2]

            marker.lifetime = rclpy.Duration(5.0)
            marker_array.markers.append(marker)

        return marker_array
