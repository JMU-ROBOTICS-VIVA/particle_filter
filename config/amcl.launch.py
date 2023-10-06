# Parts of this launch file were based turtlebot3 launch files...
# Copyright 2019 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Darby Lim

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    map_path = LaunchConfiguration(
        'map', default='/opt/ros/humble/share/turtlebot3_navigation2/map/map.yaml')

    rviz_config_dir = os.path.join(
        get_package_share_directory('particle_filter'),
        'rviz',
        'config.rviz')

    amcl_config = os.path.join(
        get_package_share_directory('particle_filter'),
        'config',
        'amcl.yaml'
    )

    return LaunchDescription([
        DeclareLaunchArgument(
            'map',
            default_value=map_path,
            description='Full path to map file to load'),
        
        DeclareLaunchArgument(
            'use_sim_time',
            default_value='true',
            description='Use simulation (Gazebo) clock if true'),

        
        Node(
            package='particle_filter',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time},
                        amcl_config],
        ),
        Node(
            package='particle_filter',
            executable='amcl_tf',
            name='amcl_tf',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
        ),


        Node(
            package='nav2_map_server',
            executable='map_server',
            name='map_server',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time, 'yaml_filename':map_path}  ],
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_dir],
            parameters=[{'use_sim_time': use_sim_time}],
            output={'both': 'log'}),

        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navigation',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time},
                        {'autostart': True},
                        {'node_names': ['map_server']}]),
            
    ])