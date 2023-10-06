"""Utility code to support a particle filter implementation.
"""
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import Pose, Transform
from builtin_interfaces.msg import Time
import tf_transformations


def pose_to_transform(pose: Pose) -> Transform:
    transform = Transform()
    transform.translation.x = pose.position.x
    transform.translation.y = pose.position.y
    transform.translation.z = pose.position.z

    transform.rotation.x = pose.orientation.x
    transform.rotation.y = pose.orientation.y
    transform.rotation.z = pose.orientation.z
    transform.rotation.w = pose.orientation.w

    return transform


def transform_to_pose(transform: Transform) -> Pose:
    pose = Pose()
    pose.position.x = transform.translation.x
    pose.position.y = transform.translation.y
    pose.position.z = transform.translation.z

    pose.orientation.x = transform.rotation.x
    pose.orientation.y = transform.rotation.y
    pose.orientation.z = transform.rotation.z
    pose.orientation.w = transform.rotation.w

    return pose


# https://github.com/ros2/geometry2/blob/5eee548ac7424f3dd9889ece279db0f3128c58b3/tf2/include/tf2/LinearMath/Transform.h#L195
def transform_inverse(transform: Transform) -> Transform:
    matrix = tf_transformations.quaternion_matrix(
        [
            transform.rotation.x,
            transform.rotation.y,
            transform.rotation.z,
            transform.rotation.w,
        ]
    )

    inv_quat = tf_transformations.quaternion_from_matrix(matrix.T)
    inv_trans = matrix[0:3, 0:3].T @ -np.array(
        [transform.translation.x, transform.translation.y, transform.translation.z]
    )

    inv_transform = Transform()
    inv_transform.rotation.x = inv_quat[0]
    inv_transform.rotation.y = inv_quat[1]
    inv_transform.rotation.z = inv_quat[2]
    inv_transform.rotation.w = inv_quat[3]
    inv_transform.translation.x = inv_trans[0]
    inv_transform.translation.y = inv_trans[1]
    inv_transform.translation.z = inv_trans[2]
    return inv_transform


def yaw_from_pose(pose):
    """Utility method to extract the yaw from a Pose object.

    Args:
       pose: geometry_msgs/Pose

    Returns:
       float: Yaw in radians
    """

    quat = np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
    )
    euler = tf_transformations.euler_from_quaternion(quat)
    return euler[2]


def pose_from_coordinates(position, quaternion):
    """Create a pose object from the provided position and angle."""
    pose = Pose()
    pose.position.x = position[0]
    pose.position.y = position[1]
    pose.position.z = position[2]
    pose.orientation.x = quaternion[0]
    pose.orientation.y = quaternion[1]
    pose.orientation.z = quaternion[2]
    pose.orientation.w = quaternion[3]
    return pose


def angle_diff(target, source):
    """Return the amount needed to change the source angle to get to the
    target angle.

    """
    return np.arctan2(np.sin(target - source), np.cos(target - source))


class Map:
    """The Map class represents an occupancy grid.

    Map entries are stored as 8-bit integers.

    Public instance variables:

        width      --  Number of columns in the occupancy grid.
        height     --  Number of rows in the occupancy grid.
        resolution --  Width of each grid square in meters.
        origin_x   --  Position of the grid cell (0,0) in
        origin_y   --    in the map coordinate system.
        grid       --  numpy array with height rows and width columns.

    Note that x increases with increasing column number and y increases
    with increasing row number.
    """

    def __init__(self, *args, **kwargs):
        """Construct an empty occupancy grid.

        Can be called -either- with a single OccupancyGrid message as
        the argument, or with any of the following provided as named
        arguments:

           keyword arguments:
                   origin_x,
                   origin_y  -- The position of grid cell (0,0) in the
                                map coordinate frame. (default -2.5, -2.5)
                   resolution-- width and height of the grid cells
                                in meters. (default .1)
                   width,
                   height    -- The grid will have height rows and width
                                columns cells.  width is the size of
                                the x-dimension and height is the size
                                of the y-dimension. (default 50, 50)

         The default arguments put (0,0) in the center of the grid.
        """

        if len(args) == 1 and isinstance(args[0], OccupancyGrid):
            self._init_from_message(args[0])
        elif len(args) == 0:
            self._init_empty(kwargs)
        else:
            raise ValueError("Constructor only supports named arguments.")

    def _init_empty(self, kwargs):
        """Set up an empty map using keyword arguments."""
        self.frame_id = "map"
        self.stamp = Time()  # There is no great way to get the actual time.
        self.origin_x = kwargs.get("origin_x", -2.5)
        self.origin_y = kwargs.get("origin_y", -2.5)
        self.width = kwargs.get("width", 50)
        self.height = kwargs.get("height", 50)
        self.resolution = kwargs.get("resolution", 0.1)
        self.grid = np.zeros((self.height, self.width))

    def _init_from_message(self, map_message):
        """
        Set up a map as an in-memory version of an OccupancyGrid message
        """
        self.frame_id = map_message.header.frame_id
        self.stamp = map_message.header.stamp
        self.width = map_message.info.width
        self.height = map_message.info.height
        self.resolution = map_message.info.resolution
        self.origin_x = map_message.info.origin.position.x
        self.origin_y = map_message.info.origin.position.y
        self.grid = np.array(map_message.data, dtype="int8").reshape(
            self.height, self.width
        )

    def to_msg(self):
        """Return a nav_msgs/OccupancyGrid representation of this map."""
        grid_msg = OccupancyGrid()
        grid_msg.header.frame_id = self.frame_id
        grid_msg.header.stamp = self.stamp
        grid_msg.info.resolution = self.resolution
        grid_msg.info.width = self.width
        grid_msg.info.height = self.height
        grid_msg.info.origin = Pose()
        grid_msg.info.origin.position.x = self.origin_x
        grid_msg.info.origin.position.y = self.origin_y
        grid_msg.info.origin.orientation.x = 0.0
        grid_msg.info.origin.orientation.y = 0.0
        grid_msg.info.origin.orientation.z = 0.0
        grid_msg.info.origin.orientation.w = 1.0

        grid_msg.data = [int(val) for val in self.grid.flatten()]
        return grid_msg

    def cell_position(self, row, col):
        """
        Determine the x, y coordinates of the center of a particular grid cell.
        """
        x = col * self.resolution + 0.5 * self.resolution + self.origin_x
        y = row * self.resolution + 0.5 * self.resolution + self.origin_y
        return x, y

    def cell_index(self, x, y):
        """
        Helper method for finding map index.  x and y are in the map
        coordinate system.
        """
        x -= self.origin_x
        y -= self.origin_y
        row = int(np.floor(y / self.resolution))
        col = int(np.floor(x / self.resolution))
        return row, col

    def set_cell(self, x, y, val):
        """
        Set the value in the grid cell containing position (x,y).
        x and y are in the map coordinate system.  No effect if (x,y)
        is out of bounds.
        """
        row, col = self.cell_index(x, y)
        try:
            if row >= 0 and col >= 0:
                self.grid[row, col] = val
        except IndexError:
            pass

    def get_cell(self, x, y):
        """
        Get the value from the grid cell containing position (x,y).
        x and y are in the map coordinate system.  Return 'nan' if
        (x,y) is out of bounds.
        """
        row, col = self.cell_index(x, y)
        try:
            if row >= 0 and col >= 0:
                return self.grid[row, col]
            else:
                return float("nan")
        except IndexError:
            return float("nan")


if __name__ == "__main__":
    transform = Transform()
    quat = tf_transformations.quaternion_from_euler(0.0, 0.0, 1.0)
    transform.rotation.x = quat[0]
    transform.rotation.y = quat[1]
    transform.rotation.z = quat[2]
    transform.rotation.w = quat[3]
    transform.translation.x = 1.0
    transform.translation.y = 2.0
    transform.translation.z = 3.0
    print(transform)
    print(transform_inverse(transform))
