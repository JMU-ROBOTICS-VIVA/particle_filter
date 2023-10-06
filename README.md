# particle_filter
ROS2 Particle Filter (Unfinished)

This is a pure-Python particle filter designed to be a more-or-less drop-in replacement for the standard [Nav2](https://github.com/ros-planning/navigation2/tree/main) localization node. 

## Included source files:

* `amcl.py` - UNFINISHED particle filter implementation.  See the docstrings in this file for a description of the topics and parameters used by this node.
* `amcl_tf.py` - Node that updates tf frames based on amcl localization.
* `particle.py` - Class representing a single particle. Implements the sensor and motion models.
* `pf_utils.py` - Utility code dealing with angles, poses, transforms etc.

## Running in Simulation
Launch the TB3 simulator:
```bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py 
```

Launch the particle filter:
```bash
ros2 launch particle_filter amcl.launch.py
```
This will start the particle filter and bring up RViz. Using the starter code, you should be able to use the "2d Pose Estimate" button to create an initial set of particles. They won't actually update until you write that code to make that happen.

You can also see the standard Nav2 particle filter in operation by running:

```bash
ros2 launch particle_filter nav2_amcl.launch.py
```

After you provide the initial pose, the Turtlebot should now be tracked correctly if you move it using teleop.

## Running On The Real Robot

To run on the real robot, launch the particle filter as follows:

```bash
ros2 launch particle_filter amcl.launch.py use_sim_time:=false map:=/path/to/the/map.yaml
```

Where `/path/to/the/map.yaml` is the path to the appropriate map file.  The `config` folder contains a map to EngGeo 1203. 
