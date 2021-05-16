import math
import pkgutil
import sys
from abc import ABC, abstractmethod
from multiprocessing.connection import Client
from pathlib import Path
from pprint import pprint

import anki_vector
import numpy as np
import pybullet
import pybullet_utils.bullet_client as bc
from scipy.ndimage import rotate as rotate_image
from scipy.ndimage.morphology import distance_transform_edt
from skimage.draw import line
from skimage.morphology import binary_dilation, dilation
from skimage.morphology.selem import disk

import vector_utils
from shortest_paths.shortest_paths import GridGraph


class VectorEnv:
    WALL_HEIGHT = 0.1
    CUBE_WIDTH = 0.044
    RECEPTACLE_WIDTH = 0.15
    IDENTITY_QUATERNION = (0, 0, 0, 1)
    REMOVED_BODY_Z = -1000  # Hide removed bodies 1000 m below
    CUBE_COLOR = (237.0 / 255, 201.0 / 255, 72.0 / 255, 1)  # Yellow
    DEBUG_LINE_COLORS = [
        (78.0 / 255, 121.0 / 255, 167.0 / 255),  # Blue
        (89.0 / 255, 169.0 / 255, 79.0 / 255),  # Green
        (176.0 / 255, 122.0 / 255, 161.0 / 255),  # Purple
        (242.0 / 255, 142.0 / 255, 43.0 / 255),  # Orange
    ]

    def __init__(
        # This comment is here to make code folding work
            self, robot_config=None, room_length=1.0, room_width=0.5, num_cubes=10, env_name='small_empty',
            use_robot_map=True, use_distance_to_receptacle_map=False, distance_to_receptacle_map_scale=0.25,
            use_shortest_path_to_receptacle_map=True, use_shortest_path_map=True, shortest_path_map_scale=0.25,
            use_intention_map=False, intention_map_encoding='ramp',
            intention_map_scale=1.0, intention_map_line_thickness=2,
            use_history_map=False,
            use_intention_channels=False, intention_channel_encoding='spatial', intention_channel_nonspatial_scale=0.025,
            use_shortest_path_partial_rewards=True, success_reward=1.0, partial_rewards_scale=2.0,
            lifting_pointless_drop_penalty=0.25, obstacle_collision_penalty=0.25, robot_collision_penalty=1.0,
            use_shortest_path_movement=True, use_partial_observations=True,
            inactivity_cutoff_per_robot=100,
            random_seed=None, use_egl_renderer=False,
            show_gui=False, show_debug_annotations=False, show_occupancy_maps=False,
            real=False, real_robot_indices=None, real_cube_indices=None, real_debug=False,
        ):

        ################################################################################
        # Arguments

        # Room configuration
        self.robot_config = robot_config
        self.room_length = room_length
        self.room_width = room_width
        self.num_cubes = num_cubes
        self.env_name = env_name

        # State representation
        self.use_robot_map = use_robot_map
        self.use_distance_to_receptacle_map = use_distance_to_receptacle_map
        self.distance_to_receptacle_map_scale = distance_to_receptacle_map_scale
        self.use_shortest_path_to_receptacle_map = use_shortest_path_to_receptacle_map
        self.use_shortest_path_map = use_shortest_path_map
        self.shortest_path_map_scale = shortest_path_map_scale
        self.use_intention_map = use_intention_map
        self.intention_map_encoding = intention_map_encoding
        self.intention_map_scale = intention_map_scale
        self.intention_map_line_thickness = intention_map_line_thickness
        self.use_history_map = use_history_map
        self.use_intention_channels = use_intention_channels
        self.intention_channel_encoding = intention_channel_encoding
        self.intention_channel_nonspatial_scale = intention_channel_nonspatial_scale

        # Rewards
        self.use_shortest_path_partial_rewards = use_shortest_path_partial_rewards
        self.success_reward = success_reward
        self.partial_rewards_scale = partial_rewards_scale
        self.lifting_pointless_drop_penalty = lifting_pointless_drop_penalty
        self.obstacle_collision_penalty = obstacle_collision_penalty
        self.robot_collision_penalty = robot_collision_penalty

        # Misc
        self.use_shortest_path_movement = use_shortest_path_movement
        self.use_partial_observations = use_partial_observations
        self.inactivity_cutoff_per_robot = inactivity_cutoff_per_robot
        self.random_seed = random_seed
        self.use_egl_renderer = use_egl_renderer

        # Debugging
        self.show_gui = show_gui
        self.show_debug_annotations = show_debug_annotations
        self.show_occupancy_maps = show_occupancy_maps

        # Real environment
        self.real = real
        self.real_robot_indices = real_robot_indices
        self.real_cube_indices = real_cube_indices
        self.real_debug = real_debug

        pprint(self.__dict__)

        ################################################################################
        # Set up pybullet

        if self.show_gui:
            self.p = bc.BulletClient(connection_mode=pybullet.GUI)
            self.p.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)
        else:
            self.p = bc.BulletClient(connection_mode=pybullet.DIRECT)
            if self.use_egl_renderer:
                assert sys.platform == 'linux'  # Linux only
                self.plugin_id = self.p.loadPlugin(pkgutil.get_loader('eglRenderer').get_filename(), "_eglRendererPlugin")

        self.p.resetDebugVisualizerCamera(
            0.47 + (5.25 - 0.47) / (10 - 0.7) * (self.room_length - 0.7), 0, -70,
            (0, -(0.07 + (1.5 - 0.07) / (10 - 0.7) * (self.room_width - 0.7)), 0))

        # Used to determine whether robot poses are out of date
        self.step_simulation_count = 0

        ################################################################################
        # Robots and room configuration

        # Random placement of robots, cubes, and obstacles
        self.room_random_state = np.random.RandomState(self.random_seed)
        self.robot_spawn_bounds = None
        self.cube_spawn_bounds = None

        # Robots
        if self.robot_config is None:
            self.robot_config = [{'lifting_robot': 1}]
        self.num_robots = sum(sum(g.values()) for g in self.robot_config)
        self.robot_group_types = [next(iter(g.keys())) for g in self.robot_config]
        self.robot_ids = None
        self.robots = None
        self.robot_groups = None
        self.robot_random_state = np.random.RandomState(self.random_seed + 1 if self.random_seed is not None else None)  # Add randomness to throwing

        # Room
        self.obstacle_ids = None
        self.cube_ids = None
        self.receptacle_id = None
        if not any('rescue_robot' in g for g in self.robot_config):
            self.receptacle_position = (self.room_length / 2 - VectorEnv.RECEPTACLE_WIDTH / 2, self.room_width / 2 - VectorEnv.RECEPTACLE_WIDTH / 2, 0)

        # Collections for keeping track of environment state
        self.obstacle_collision_body_b_ids_set = None  # For collision detection
        self.robot_collision_body_b_ids_set = None  # For collision detection
        self.available_cube_ids_set = None  # Excludes removed cubes, and cubes that are being lifted, thrown, or rescued
        self.removed_cube_ids_set = None  # Cubes that have been removed

        ################################################################################
        # Misc

        # End an episode after too many steps of inactivity
        self.inactivity_cutoff = self.num_robots * self.inactivity_cutoff_per_robot

        # Stats
        self.steps = None
        self.simulation_steps = None
        self.inactivity_steps = None

        ################################################################################
        # Real environment

        if self.real:
            assert len(self.real_robot_indices) == self.num_robots
            assert len(self.real_cube_indices) == self.num_cubes
            self.real_robot_indices_map = None
            self.real_cube_indices_map = None

            # Connect to aruco server for pose estimates
            address = 'localhost'
            if self.env_name.startswith('large'):
                # Left camera, right camera
                self.conns = [Client((address, 6001), authkey=b'secret password'), Client((address, 6002), authkey=b'secret password')]
            else:
                self.conns = [Client((address, 6000), authkey=b'secret password')]

    def reset(self):
        # Disconnect robots
        if self.real:
            self._disconnect_robots()

        # Reset pybullet
        self.p.resetSimulation()
        self.p.setRealTimeSimulation(0)
        self.p.setGravity(0, 0, -9.8)

        # Create env
        self._create_env()
        if self.real:
            self.real_robot_indices_map = dict(zip(self.robot_ids, self.real_robot_indices))
            self.real_cube_indices_map = dict(zip(self.cube_ids, self.real_cube_indices))

        # Reset poses
        if self.real:
            self.update_poses()
        else:
            self._reset_poses()
        self._step_simulation_until_still()

        # Set awaiting new action for first robot
        self._set_awaiting_new_action()

        # State representation
        for robot in self.robots:
            robot.update_map()

        # Stats
        self.steps = 0
        self.simulation_steps = 0
        self.inactivity_steps = 0

        return self.get_state()

    def store_new_action(self, action):
        for robot_group, robot_group_actions in zip(self.robot_groups, action):
            for robot, a in zip(robot_group, robot_group_actions):
                if a is not None:
                    robot.store_new_action(a)

    def step(self, action):
        ################################################################################
        # Setup before action execution

        self.store_new_action(action)

        # Store initial cube positions for pushing partial rewards
        if any(isinstance(robot, PushingRobot) for robot in self.robots):
            initial_cube_positions = {}
            for cube_id in self.available_cube_ids_set:
                initial_cube_positions[cube_id] = self.get_cube_position(cube_id)

        ################################################################################
        # Execute actions

        if self.real:
            sim_steps = self._execute_actions_real()
        else:
            sim_steps = self._execute_actions()
        self._set_awaiting_new_action()

        ################################################################################
        # Process cubes after action execution

        for cube_id in self.available_cube_ids_set.copy():
            cube_position = self.get_cube_position(cube_id)

            # Reset out-of-bounds cubes
            if (cube_position[2] > VectorEnv.WALL_HEIGHT + 0.49 * VectorEnv.CUBE_WIDTH or  # On top of obstacle
                    cube_position[2] < 0.4 * VectorEnv.CUBE_WIDTH):  # Inside obstacle (0.4 since dropped cubes can temporarily go into the ground)
                pos_x, pos_y, heading = self._get_random_cube_pose()
                self.reset_cube_pose(cube_id, pos_x, pos_y, heading)
                continue

            if self.receptacle_id is not None:
                closest_robot = self.robots[np.argmin([distance(robot.get_position(), cube_position) for robot in self.robots])]

                # Process final cube position for pushing partial rewards
                if isinstance(closest_robot, PushingRobot):
                    closest_robot.process_cube_position(cube_id, initial_cube_positions)

                # Process cubes that are in the receptacle (cubes were pushed in)
                if self.cube_position_in_receptacle(cube_position):
                    closest_robot.process_cube_success()
                    self.remove_cube(cube_id)
                    self.available_cube_ids_set.remove(cube_id)

        # Robots that are awaiting new action need an up-to-date map
        for robot in self.robots:
            if robot.awaiting_new_action:
                robot.update_map()

        ################################################################################
        # Compute rewards and stats

        # Increment counters
        self.steps += 1
        self.simulation_steps += sim_steps
        if sum(robot.cubes for robot in self.robots) > 0:
            self.inactivity_steps = 0
        else:
            self.inactivity_steps += 1

        # Episode ends after too many steps of inactivity
        done = len(self.removed_cube_ids_set) == self.num_cubes or self.inactivity_steps >= self.inactivity_cutoff

        # Compute per-robot rewards and stats
        for robot in self.robots:
            if robot.awaiting_new_action or done:
                robot.compute_rewards_and_stats(done=done)

        ################################################################################
        # Compute items to return

        state = [[None for _ in g] for g in self.robot_groups] if done else self.get_state()
        reward = [[robot.reward if (robot.awaiting_new_action or done) else None for robot in robot_group] for robot_group in self.robot_groups]
        info = {
            'steps': self.steps,
            'simulation_steps': self.simulation_steps,
            'distance': [[robot.distance if (robot.awaiting_new_action or done) else None for robot in g] for g in self.robot_groups],
            'cumulative_cubes': [[robot.cumulative_cubes if (robot.awaiting_new_action or done) else None for robot in g] for g in self.robot_groups],
            'cumulative_distance': [[robot.cumulative_distance if (robot.awaiting_new_action or done) else None for robot in g] for g in self.robot_groups],
            'cumulative_reward': [[robot.cumulative_reward if (robot.awaiting_new_action or done) else None for robot in g] for g in self.robot_groups],
            'cumulative_obstacle_collisions': [[robot.cumulative_obstacle_collisions if (robot.awaiting_new_action or done) else None for robot in g] for g in self.robot_groups],
            'cumulative_robot_collisions': [[robot.cumulative_robot_collisions if (robot.awaiting_new_action or done) else None for robot in g] for g in self.robot_groups],
            'total_cubes': sum(robot.cumulative_cubes for robot in self.robots),
            'total_obstacle_collisions': sum(robot.cumulative_obstacle_collisions for robot in self.robots),
            'total_robot_collisions': sum(robot.cumulative_robot_collisions for robot in self.robots),
        }

        return state, reward, done, info

    def get_state(self, all_robots=False, save_figures=False):
        return [[robot.get_state(save_figures=save_figures) if robot.awaiting_new_action or all_robots else None for robot in robot_group] for robot_group in self.robot_groups]

    def close(self):
        if not self.show_gui and self.use_egl_renderer:
            self.p.unloadPlugin(self.plugin_id)
        self.p.disconnect()
        if self.real:
            self._disconnect_robots()

    def step_simulation(self):
        self.p.stepSimulation()
        #import time; time.sleep(1.0 / 60)
        self.step_simulation_count += 1

    def get_cube_pose(self, cube_id):
        return self.p.getBasePositionAndOrientation(cube_id)

    def get_cube_position(self, cube_id):
        position, _ = self.get_cube_pose(cube_id)
        return position

    def reset_cube_pose(self, cube_id, pos_x, pos_y, heading):
        position = (pos_x, pos_y, VectorEnv.CUBE_WIDTH / 2)
        self.p.resetBasePositionAndOrientation(cube_id, position, heading_to_orientation(heading))

    def remove_cube(self, cube_id):
        self.p.resetBasePositionAndOrientation(cube_id, (0, 0, VectorEnv.REMOVED_BODY_Z), VectorEnv.IDENTITY_QUATERNION)
        self.removed_cube_ids_set.add(cube_id)

    def cube_position_in_receptacle(self, cube_position):
        assert self.receptacle_id is not None

        half_width = (VectorEnv.RECEPTACLE_WIDTH - VectorEnv.CUBE_WIDTH) / 2
        #if (self.receptacle_position[0] - half_width < cube_position[0] < self.receptacle_position[0] + half_width and
        #        self.receptacle_position[1] - half_width < cube_position[1] < self.receptacle_position[1] + half_width):
        if cube_position[0] > self.receptacle_position[0] - half_width and cube_position[1] > self.receptacle_position[1] - half_width:
            # Note: Assumes receptacle is in top right corner
            return True
        return False

    def get_robot_group_types(self):
        return self.robot_group_types

    @staticmethod
    def get_state_width():
        return Mapper.LOCAL_MAP_PIXEL_WIDTH

    @staticmethod
    def get_num_output_channels(robot_type):
        return Robot.get_robot_cls(robot_type).NUM_OUTPUT_CHANNELS

    @staticmethod
    def get_action_space(robot_type):
        return VectorEnv.get_num_output_channels(robot_type) * Mapper.LOCAL_MAP_PIXEL_WIDTH * Mapper.LOCAL_MAP_PIXEL_WIDTH

    def get_camera_image(self, image_width=1024, image_height=768):
        renderer = pybullet.ER_BULLET_HARDWARE_OPENGL if self.show_gui else pybullet.ER_TINY_RENDERER
        return self.p.getCameraImage(image_width, image_height, flags=pybullet.ER_NO_SEGMENTATION_MASK, renderer=renderer)[2]

    def start_video_logging(self, video_path):
        assert self.show_gui
        return self.p.startStateLogging(pybullet.STATE_LOGGING_VIDEO_MP4, video_path)

    def stop_video_logging(self, log_id):
        self.p.stopStateLogging(log_id)

    def update_poses(self):
        assert self.real

        # Get new pose estimates
        for conn in self.conns:
            if self.real_debug:
                debug_data = [(robot.waypoint_positions, robot.target_end_effector_position, robot.controller.debug_data) for robot in self.robots]
                #debug_data = [(robot.controller.get_intention_path(), robot.target_end_effector_position, robot.controller.debug_data) for robot in self.robots]
                #debug_data = [(robot.controller.get_history_path(), robot.target_end_effector_position, robot.controller.debug_data) for robot in self.robots]
                conn.send(debug_data)
            else:
                conn.send(None)

        for conn in self.conns:
            robot_poses, cube_poses = conn.recv()

            # Update cube poses
            if cube_poses is not None:
                for cube_id in self.available_cube_ids_set:
                    cube_pose = cube_poses.get(self.real_cube_indices_map[cube_id], None)
                    if cube_pose is not None:
                        self.reset_cube_pose(cube_id, cube_pose['position'][0], cube_pose['position'][1], cube_pose['heading'])

            for robot in self.robots:
                # Update robot poses
                if robot_poses is not None:
                    robot_pose = robot_poses.get(self.real_robot_indices_map[robot.id], None)
                    if robot_pose is not None:
                        robot.reset_pose(robot_pose['position'][0], robot_pose['position'][1], robot_pose['heading'])

                if cube_poses is not None:
                    if isinstance(robot, (LiftingRobot, ThrowingRobot, RescueRobot)) and robot.cube_id is not None:
                        cube_pose = cube_poses.get(self.real_cube_indices_map[robot.cube_id])

                        if cube_pose is not None:
                            if isinstance(robot, LiftingRobot):
                                robot.controller.monitor_lifted_cube(cube_pose)
                            elif isinstance(robot, ThrowingRobot):
                                self.reset_cube_pose(robot.cube_id, cube_pose['position'][0], cube_pose['position'][1], cube_pose['heading'])

                        if isinstance(robot, RescueRobot):
                            robot.controller.monitor_rescued_cube(cube_pose)

        self.step_simulation()

    def _create_env(self):
        # Assertions
        assert self.room_length >= self.room_width
        assert self.num_cubes > 0
        assert all(len(g) == 1 for g in self.robot_config)  # Each robot group should be homogeneous
        assert not len(self.robot_group_types) > 4  # More than 4 groups not supported
        if any('rescue_robot' in g for g in self.robot_config):
            assert all(robot_type == 'rescue_robot' for g in self.robot_config for robot_type in g)

        # Create floor
        floor_thickness = 10
        wall_thickness = 1.4
        room_length_with_walls = self.room_length + 2 * wall_thickness
        room_width_with_walls = self.room_width + 2 * wall_thickness
        floor_half_extents = (room_length_with_walls / 2, room_width_with_walls / 2, floor_thickness / 2)
        floor_collision_shape_id = self.p.createCollisionShape(pybullet.GEOM_BOX, halfExtents=floor_half_extents)
        floor_visual_shape_id = self.p.createVisualShape(pybullet.GEOM_BOX, halfExtents=floor_half_extents)
        self.p.createMultiBody(0, floor_collision_shape_id, floor_visual_shape_id, (0, 0, -floor_thickness / 2))

        # Create obstacles (including walls)
        obstacle_color = (0.9, 0.9, 0.9, 1)
        rounded_corner_path = str(Path(__file__).parent / 'assets' / 'rounded_corner.obj')
        self.obstacle_ids = []
        for obstacle in self._get_obstacles(wall_thickness):
            if obstacle['type'] == 'corner':
                obstacle_collision_shape_id = self.p.createCollisionShape(pybullet.GEOM_MESH, fileName=rounded_corner_path)
                obstacle_visual_shape_id = self.p.createVisualShape(pybullet.GEOM_MESH, fileName=rounded_corner_path, rgbaColor=obstacle_color)
            else:
                half_height = VectorEnv.CUBE_WIDTH / 2 if 'low' in obstacle else VectorEnv.WALL_HEIGHT / 2
                obstacle_half_extents = (obstacle['x_len'] / 2, obstacle['y_len'] / 2, half_height)
                obstacle_collision_shape_id = self.p.createCollisionShape(pybullet.GEOM_BOX, halfExtents=obstacle_half_extents)
                obstacle_visual_shape_id = self.p.createVisualShape(pybullet.GEOM_BOX, halfExtents=obstacle_half_extents, rgbaColor=obstacle_color)

            obstacle_id = self.p.createMultiBody(
                0, obstacle_collision_shape_id, obstacle_visual_shape_id,
                (obstacle['position'][0], obstacle['position'][1], VectorEnv.WALL_HEIGHT / 2), heading_to_orientation(obstacle['heading']))
            self.obstacle_ids.append(obstacle_id)

        # Create target receptacle
        if not any('rescue_robot' in g for g in self.robot_config):
            receptacle_color = (1, 87.0 / 255, 89.0 / 255, 1)  # Red
            receptacle_collision_shape_id = self.p.createCollisionShape(pybullet.GEOM_BOX, halfExtents=(0, 0, 0))
            receptacle_visual_shape_id = self.p.createVisualShape(
                #pybullet.GEOM_BOX, halfExtents=(VectorEnv.RECEPTACLE_WIDTH / 2, VectorEnv.RECEPTACLE_WIDTH / 2, 0),  # Gets rendered incorrectly in EGL renderer if height is 0
                pybullet.GEOM_BOX, halfExtents=(VectorEnv.RECEPTACLE_WIDTH / 2, VectorEnv.RECEPTACLE_WIDTH / 2, 0.0001),
                rgbaColor=receptacle_color, visualFramePosition=(0, 0, 0.0001))
            self.receptacle_id = self.p.createMultiBody(0, receptacle_collision_shape_id, receptacle_visual_shape_id, self.receptacle_position)

        # Create robots
        self.robot_collision_body_b_ids_set = set()
        self.robot_ids = []
        self.robots = []  # Flat list
        self.robot_groups = [[] for _ in range(len(self.robot_config))]  # Grouped list
        for robot_group_index, g in enumerate(self.robot_config):
            robot_type, count = next(iter(g.items()))
            for _ in range(count):
                if self.real:
                    real_robot_index = self.real_robot_indices[len(self.robots)]
                    robot = Robot.get_robot(robot_type, self, robot_group_index, real=True, real_robot_index=real_robot_index)
                else:
                    robot = Robot.get_robot(robot_type, self, robot_group_index)
                self.robots.append(robot)
                self.robot_groups[robot_group_index].append(robot)
                self.robot_ids.append(robot.id)

        # Create cubes
        cube_half_extents = (VectorEnv.CUBE_WIDTH / 2, VectorEnv.CUBE_WIDTH / 2, VectorEnv.CUBE_WIDTH / 2)
        cube_collision_shape_id = self.p.createCollisionShape(pybullet.GEOM_BOX, halfExtents=cube_half_extents)
        cube_visual_shape_id = self.p.createVisualShape(pybullet.GEOM_BOX, halfExtents=cube_half_extents, rgbaColor=VectorEnv.CUBE_COLOR)
        cube_mass = 0.024  # 24 g
        self.cube_ids = []
        for _ in range(self.num_cubes):
            cube_id = self.p.createMultiBody(cube_mass, cube_collision_shape_id, cube_visual_shape_id)
            self.cube_ids.append(cube_id)

        # Initialize collections
        self.obstacle_collision_body_b_ids_set = set(self.obstacle_ids)
        self.robot_collision_body_b_ids_set.update(self.robot_ids)
        self.available_cube_ids_set = set(self.cube_ids)
        self.removed_cube_ids_set = set()

    def _get_obstacles(self, wall_thickness):
        if self.env_name.startswith('small'):
            assert math.isclose(self.room_length, 1)
            assert math.isclose(self.room_width, 0.5)
        elif self.env_name.startswith('large'):
            assert math.isclose(self.room_length, 1)
            assert math.isclose(self.room_width, 1)

        def add_divider(x_offset=0):
            divider_width = 0.05
            opening_width = 0.16
            obstacles.append({'type': 'divider', 'position': (x_offset, 0), 'heading': 0, 'x_len': divider_width, 'y_len': self.room_width - 2 * opening_width})
            self.robot_spawn_bounds = (x_offset + divider_width / 2, None, None, None)
            self.cube_spawn_bounds = (None, x_offset - divider_width / 2, None, None)

        def add_tunnels(tunnel_length, x_offset=0, y_offset=0):
            tunnel_width = 0.18
            tunnel_x = (self.room_length + tunnel_width) / 6 + x_offset
            outer_divider_len = self.room_length / 2 - tunnel_x - tunnel_width / 2
            divider_x = self.room_length / 2 - outer_divider_len / 2
            middle_divider_len = 2 * (tunnel_x - tunnel_width / 2)
            obstacles.append({'type': 'divider', 'position': (-divider_x, y_offset), 'heading': 0, 'x_len': outer_divider_len, 'y_len': tunnel_length})
            obstacles.append({'type': 'divider', 'position': (0, y_offset), 'heading': 0, 'x_len': middle_divider_len, 'y_len': tunnel_length})
            obstacles.append({'type': 'divider', 'position': (divider_x, y_offset), 'heading': 0, 'x_len': outer_divider_len, 'y_len': tunnel_length})
            self.robot_spawn_bounds = (None, None, y_offset + tunnel_length / 2, None)
            self.cube_spawn_bounds = (None, None, None, y_offset - tunnel_length / 2)

        def add_rooms(x_offset=0, y_offset=0):
            divider_width = 0.05
            opening_width = 0.18
            divider_len = self.room_width / 2 - opening_width - divider_width / 2
            top_divider_len = divider_len - y_offset
            bot_divider_len = divider_len + y_offset
            top_divider_y = self.room_width / 2 - opening_width - top_divider_len / 2
            bot_divider_y = -self.room_width / 2 + opening_width + bot_divider_len / 2
            obstacles.append({'type': 'divider', 'position': (0, y_offset), 'heading': 0, 'x_len': self.room_length - 2 * opening_width, 'y_len': divider_width})
            obstacles.append({'type': 'divider', 'position': (x_offset, top_divider_y), 'heading': 0, 'x_len': divider_width, 'y_len': top_divider_len, 'snap_y': y_offset + divider_width / 2})
            obstacles.append({'type': 'divider', 'position': (x_offset, bot_divider_y), 'heading': 0, 'x_len': divider_width, 'y_len': bot_divider_len, 'snap_y': y_offset - divider_width / 2})

        # Walls
        obstacles = []
        for x, y, length, width in [
                (-self.room_length / 2 - wall_thickness / 2, 0, wall_thickness, self.room_width),
                (self.room_length / 2 + wall_thickness / 2, 0, wall_thickness, self.room_width),
                (0, -self.room_width / 2 - wall_thickness / 2, self.room_length + 2 * wall_thickness, wall_thickness),
                (0, self.room_width / 2 + wall_thickness / 2, self.room_length + 2 * wall_thickness, wall_thickness),
            ]:
            obstacles.append({'type': 'wall', 'position': (x, y), 'heading': 0, 'x_len': length, 'y_len': width})

        # Other obstacles
        if self.env_name == 'small_empty':
            pass

        elif self.env_name == 'small_divider_norand':
            add_divider()

        elif self.env_name == 'small_divider':
            add_divider(x_offset=self.room_random_state.uniform(-0.1, 0.1))

        elif self.env_name == 'large_empty':
            pass

        elif self.env_name == 'large_doors_norand':
            add_tunnels(0.05)

        elif self.env_name == 'large_doors':
            add_tunnels(0.05, x_offset=self.room_random_state.uniform(-0.05, 0.05), y_offset=self.room_random_state.uniform(-0.1, 0.1))

        elif self.env_name == 'large_tunnels_norand':
            add_tunnels(0.25)

        elif self.env_name == 'large_tunnels':
            add_tunnels(0.25, x_offset=self.room_random_state.uniform(-0.05, 0.05), y_offset=self.room_random_state.uniform(-0.05, 0.05))

        elif self.env_name == 'large_rooms_norand':
            add_rooms()

        elif self.env_name == 'large_rooms':
            add_rooms(x_offset=self.room_random_state.uniform(-0.05, 0.05), y_offset=self.room_random_state.uniform(-0.05, 0.05))

        else:
            raise Exception(self.env_name)

        ################################################################################
        # Rounded corners

        rounded_corner_width = 0.1006834873
        # Room corners
        for i, (x, y) in enumerate([
                (-self.room_length / 2, self.room_width / 2),
                (self.room_length / 2, self.room_width / 2),
                (self.room_length / 2, -self.room_width / 2),
                (-self.room_length / 2, -self.room_width / 2),
            ]):
            if any('rescue_robot' in g for g in self.robot_config) or distance((x, y), self.receptacle_position) > (1 + 1e-6) * (VectorEnv.RECEPTACLE_WIDTH / 2) * math.sqrt(2):
                heading = -math.radians(i * 90)
                offset = rounded_corner_width / math.sqrt(2)
                adjusted_position = (x + offset * math.cos(heading - math.radians(45)), y + offset * math.sin(heading - math.radians(45)))
                obstacles.append({'type': 'corner', 'position': adjusted_position, 'heading': heading})

        # Corners between walls and dividers
        new_obstacles = []
        for obstacle in obstacles:
            if obstacle['type'] == 'divider':
                position, length, width = obstacle['position'], obstacle['x_len'], obstacle['y_len']
                x, y = position
                corner_positions = None
                if math.isclose(x - length / 2, -self.room_length / 2):
                    corner_positions = [(-self.room_length / 2, y - width / 2), (-self.room_length / 2, y + width / 2)]
                    corner_headings = [0, 90]
                elif math.isclose(x + length / 2, self.room_length / 2):
                    corner_positions = [(self.room_length / 2, y - width / 2), (self.room_length / 2, y + width / 2)]
                    corner_headings = [-90, 180]
                elif math.isclose(y - width / 2, -self.room_width / 2):
                    corner_positions = [(x - length / 2, -self.room_width / 2), (x + length / 2, -self.room_width / 2)]
                    corner_headings = [180, 90]
                elif math.isclose(y + width / 2, self.room_width / 2):
                    corner_positions = [(x - length / 2, self.room_width / 2), (x + length / 2, self.room_width / 2)]
                    corner_headings = [-90, 0]
                elif 'snap_y' in obstacle:
                    snap_y = obstacle['snap_y']
                    corner_positions = [(x - length / 2, snap_y), (x + length / 2, snap_y)]
                    corner_headings = [-90, 0] if snap_y > y else [180, 90]
                if corner_positions is not None:
                    for position, heading in zip(corner_positions, corner_headings):
                        heading = math.radians(heading)
                        offset = rounded_corner_width / math.sqrt(2)
                        adjusted_position = (
                            position[0] + offset * math.cos(heading - math.radians(45)),
                            position[1] + offset * math.sin(heading - math.radians(45))
                        )
                        obstacles.append({'type': 'corner', 'position': adjusted_position, 'heading': heading})
        obstacles.extend(new_obstacles)

        return obstacles

    def _reset_poses(self):
        # Reset robot poses
        for robot in self.robots:
            pos_x, pos_y, heading = self._get_random_robot_pose(padding=robot.RADIUS, bounds=self.robot_spawn_bounds)
            robot.reset_pose(pos_x, pos_y, heading)

        # Reset cube poses
        for cube_id in self.cube_ids:
            pos_x, pos_y, heading = self._get_random_cube_pose()
            self.reset_cube_pose(cube_id, pos_x, pos_y, heading)

        # Check if any robots need another pose reset
        done = False
        while not done:
            done = True
            self.step_simulation()
            for robot in self.robots:
                reset_robot_pose = False

                # Check if robot is stacked on top of a cube
                if robot.get_position(set_z_to_zero=False)[2] > 0.001:  # 1 mm
                    reset_robot_pose = True

                # Check if robot is inside an obstacle or another robot
                for contact_point in self.p.getContactPoints(robot.id):
                    if contact_point[2] in self.obstacle_collision_body_b_ids_set or contact_point[2] in self.robot_collision_body_b_ids_set:
                        reset_robot_pose = True
                        break

                if reset_robot_pose:
                    done = False
                    pos_x, pos_y, heading = self._get_random_robot_pose(padding=robot.RADIUS, bounds=self.robot_spawn_bounds)
                    robot.reset_pose(pos_x, pos_y, heading)

    def _get_random_cube_pose(self):
        done = False
        while not done:
            pos_x, pos_y = self._get_random_position(padding=VectorEnv.CUBE_WIDTH / 2, bounds=self.cube_spawn_bounds)

            # Only spawn cubes outside of the receptacle
            if self.receptacle_id is None or not self.cube_position_in_receptacle((pos_x, pos_y)):
                done = True
        heading = self.room_random_state.uniform(-math.pi, math.pi)
        return pos_x, pos_y, heading

    def _get_random_robot_pose(self, padding=0, bounds=None):
        position_x, position_y = self._get_random_position(padding=padding, bounds=bounds)
        heading = self.room_random_state.uniform(-math.pi, math.pi)
        return position_x, position_y, heading

    def _get_random_position(self, padding=0, bounds=None):
        low_x = -self.room_length / 2 + padding
        high_x = self.room_length / 2 - padding
        low_y = -self.room_width / 2 + padding
        high_y = self.room_width / 2 - padding
        if bounds is not None:
            x_min, x_max, y_min, y_max = bounds
            if x_min is not None:
                low_x = x_min + padding
            if x_max is not None:
                high_x = x_max - padding
            if y_min is not None:
                low_y = y_min + padding
            if y_max is not None:
                high_y = y_max - padding
        position_x, position_y = self.room_random_state.uniform((low_x, low_y), (high_x, high_y))
        return position_x, position_y

    def _step_simulation_until_still(self):
        # Kick-start gravity
        for _ in range(2):
            self.step_simulation()

        movable_body_ids = self.robot_ids + self.cube_ids
        prev_positions = []
        sim_steps = 0
        done = False
        while not done:
            # Check whether any bodies moved since last step
            positions = [self.p.getBasePositionAndOrientation(body_id)[0] for body_id in movable_body_ids]
            if len(prev_positions) > 0:
                done = True
                for prev_position, position in zip(prev_positions, positions):
                    change = distance(prev_position, position)
                    # Ignore removed cubes (negative z)
                    if position[2] > -0.0001 and change > 0.0005:  # 0.5 mm
                        done = False
                        break
            prev_positions = positions

            self.step_simulation()
            sim_steps += 1

            if sim_steps > 800:
                break

    def _set_awaiting_new_action(self):
        if sum(robot.awaiting_new_action for robot in self.robots) == 0:
            for robot in self.robots:
                if robot.is_idle():
                    robot.awaiting_new_action = True
                    break

    def _execute_actions(self):
        sim_steps = 0
        while True:
            if any(robot.is_idle() for robot in self.robots):
                break

            self.step_simulation()
            sim_steps += 1
            for robot in self.robots:
                robot.step()

        return sim_steps

    def _execute_actions_real(self):
        assert self.real

        # If debug mode is enabled, all robots will pause and resume actions during any robot's action selection
        if self.real_debug:
            for robot in self.robots:
                robot.controller.resume()

        sim_steps = 0
        any_idle = False
        while True:
            if not any_idle and any(robot.is_idle() for robot in self.robots):
                any_idle = True
                if self.real_debug:
                    for robot in self.robots:
                        robot.controller.pause()

            if any_idle:
                # If debug mode is enabled, do not exit loop until all robots have actually stopped moving
                if not self.real_debug or all((robot.is_idle() or robot.controller.state == 'paused') for robot in self.robots):
                    break

            self.update_poses()
            sim_steps += 1
            for robot in self.robots:
                robot.step()

        return sim_steps

    def _disconnect_robots(self):
        assert self.real
        if self.robots is not None:
            for robot in self.robots:
                robot.controller.disconnect()

class Robot(ABC):
    HALF_WIDTH = 0.03
    BACKPACK_OFFSET = -0.0135
    BASE_LENGTH = 0.065  # Does not include the hooks
    TOP_LENGTH = 0.057  # Leaves 1 mm gap for lifted cube
    END_EFFECTOR_LOCATION = BACKPACK_OFFSET + BASE_LENGTH
    RADIUS = math.sqrt(HALF_WIDTH**2 + END_EFFECTOR_LOCATION**2)
    HEIGHT = 0.07
    NUM_OUTPUT_CHANNELS = 1
    COLOR = (0.3529, 0.3529, 0.3529, 1)  # Gray
    CONSTRAINT_MAX_FORCE = 10

    @abstractmethod  # Should not be instantiated directly
    def __init__(self, env, group_index, real=False, real_robot_index=None):
        self.env = env
        self.group_index = group_index
        self.real = real
        self.id = self._create_multi_body()
        self.cid = self.env.p.createConstraint(self.id, -1, -1, -1, pybullet.JOINT_FIXED, None, (0, 0, 0), (0, 0, 0))
        self._last_step_simulation_count = -1  # Used to determine whether pose is out of date
        self._position_raw = None  # Most current position, not to be directly accessed (use self.get_position())
        self._position = None  # Most current position (with z set to 0), not to be directly accessed (use self.get_position())
        self._heading = None  # Most current heading, not to be directly accessed (use self.get_heading())

        # Movement
        self.action = None
        self.target_end_effector_position = None
        self.waypoint_positions = None
        self.waypoint_headings = None
        self.controller = RealRobotController(self, real_robot_index, debug=self.env.real_debug) if real else RobotController(self)

        # Collision detection
        self.collision_body_a_ids_set = set([self.id])

        # State representation
        self.mapper = Mapper(self.env, self)

        # Step variables and stats
        self.awaiting_new_action = False  # Only one robot at a time can be awaiting new action
        self.cubes = 0
        self.reward = None
        self.cubes_with_reward = 0
        self.distance = 0
        self.prev_waypoint_position = None  # For tracking distance traveled over the step
        self.collided_with_obstacle = False
        self.collided_with_robot = False

        # Episode stats (robots are recreated every episode)
        self.cumulative_cubes = 0
        self.cumulative_distance = 0
        self.cumulative_reward = 0
        self.cumulative_obstacle_collisions = 0
        self.cumulative_robot_collisions = 0

    def store_new_action(self, action):
        # Action is specified as an index specifying an end effector action, along with (row, col) of the selected pixel location
        self.action = tuple(np.unravel_index(action, (self.NUM_OUTPUT_CHANNELS, Mapper.LOCAL_MAP_PIXEL_WIDTH, Mapper.LOCAL_MAP_PIXEL_WIDTH)))  # Immutable tuple

        # Get current robot pose
        current_position, current_heading = self.get_position(), self.get_heading()

        # Compute distance from front of robot (not center of robot), which is used to find the
        # robot position and heading that would place the end effector over the specified location
        dx, dy = Mapper.pixel_indices_to_position(self.action[1], self.action[2], (Mapper.LOCAL_MAP_PIXEL_WIDTH, Mapper.LOCAL_MAP_PIXEL_WIDTH))
        dist = math.sqrt(dx**2 + dy**2)
        theta = current_heading + math.atan2(-dx, dy)
        self.target_end_effector_position = (current_position[0] + dist * math.cos(theta), current_position[1] + dist * math.sin(theta), 0)

        ################################################################################
        # Waypoints

        # Compute waypoint positions
        if self.env.use_shortest_path_movement:
            self.waypoint_positions = self.mapper.shortest_path(current_position, self.target_end_effector_position)
        else:
            self.waypoint_positions = [current_position, self.target_end_effector_position]

        # Compute waypoint headings
        self.waypoint_headings = [current_heading]
        for i in range(1, len(self.waypoint_positions)):
            dx = self.waypoint_positions[i][0] - self.waypoint_positions[i - 1][0]
            dy = self.waypoint_positions[i][1] - self.waypoint_positions[i - 1][1]
            self.waypoint_headings.append(restrict_heading_range(math.atan2(dy, dx)))

        # Compute target position and heading for the robot. This involves applying an
        # offset to shift the final waypoint from end effector position to robot position.
        signed_dist = distance(self.waypoint_positions[-2], self.waypoint_positions[-1]) - (self.END_EFFECTOR_LOCATION + VectorEnv.CUBE_WIDTH / 2)
        target_heading = self.waypoint_headings[-1]
        target_position = (
            self.waypoint_positions[-2][0] + signed_dist * math.cos(target_heading),
            self.waypoint_positions[-2][1] + signed_dist * math.sin(target_heading),
            0
        )
        self.waypoint_positions[-1] = target_position

        # Avoid awkward backing up to reach the last waypoint
        if len(self.waypoint_positions) > 2 and signed_dist < 0:
            self.waypoint_positions[-2] = self.waypoint_positions[-1]
            dx = self.waypoint_positions[-2][0] - self.waypoint_positions[-3][0]
            dy = self.waypoint_positions[-2][1] - self.waypoint_positions[-3][1]
            self.waypoint_headings[-2] = restrict_heading_range(math.atan2(dy, dx))

        ################################################################################
        # Step variables and stats

        # Reset controller
        self.controller.reset()
        self.controller.new_action()

        # Reset step variables and stats
        self.awaiting_new_action = False
        self.cubes = 0
        self.reward = None
        self.cubes_with_reward = 0
        self.distance = 0
        self.prev_waypoint_position = current_position
        self.collided_with_obstacle = False
        self.collided_with_robot = False

    def step(self):
        self.controller.step()

    def update_map(self):
        self.mapper.update()

    def get_state(self, save_figures=False):
        return self.mapper.get_state(save_figures=save_figures)

    def process_cube_success(self):
        self.cubes += 1

    def compute_rewards_and_stats(self, done=False):
        # Ways a step can end
        # - Successfully completed action
        # - Collision
        # - Step limit exceeded
        # - Episode ended (no cubes left or too many steps of inactivity)

        if done:
            self.update_distance()
            self.controller.reset()

        # Calculate final reward
        success_reward = self.env.success_reward * self.cubes_with_reward
        obstacle_collision_penalty = -self.env.obstacle_collision_penalty * self.collided_with_obstacle
        robot_collision_penalty = -self.env.robot_collision_penalty * self.collided_with_robot
        self.reward = success_reward + obstacle_collision_penalty + robot_collision_penalty

        # Update cumulative stats
        self.cumulative_cubes += self.cubes
        self.cumulative_reward += self.reward
        self.cumulative_distance += self.distance
        self.cumulative_obstacle_collisions += self.collided_with_obstacle
        self.cumulative_robot_collisions += self.collided_with_robot

    def reset(self):
        self.action = None
        self.target_end_effector_position = None
        self.waypoint_positions = None
        self.waypoint_headings = None
        self.controller.reset()

    def is_idle(self):
        return self.controller.state == 'idle'

    def get_position(self, set_z_to_zero=True):
        # Returned position is immutable tuple
        if self._last_step_simulation_count < self.env.step_simulation_count:
            self._update_pose()
        if not set_z_to_zero:
            return self._position_raw
        return self._position

    def get_heading(self):
        if self._last_step_simulation_count < self.env.step_simulation_count:
            self._update_pose()
        return self._heading

    def reset_pose(self, position_x, position_y, heading):
        # Reset robot pose
        position = (position_x, position_y, 0)
        orientation = heading_to_orientation(heading)
        self.env.p.resetBasePositionAndOrientation(self.id, position, orientation)
        self.env.p.changeConstraint(self.cid, jointChildPivot=position, jointChildFrameOrientation=orientation, maxForce=Robot.CONSTRAINT_MAX_FORCE)
        self._last_step_simulation_count = -1

    def check_for_collisions(self):
        for body_a_id in self.collision_body_a_ids_set:
            for contact_point in self.env.p.getContactPoints(body_a_id):
                body_b_id = contact_point[2]
                if body_b_id in self.collision_body_a_ids_set:
                    continue
                if body_b_id in self.env.obstacle_collision_body_b_ids_set:
                    self.collided_with_obstacle = True
                if body_b_id in self.env.robot_collision_body_b_ids_set:
                    self.collided_with_robot = True
                if self.collided_with_obstacle or self.collided_with_robot:
                    break

    def update_distance(self):
        current_position = self.get_position()
        self.distance += distance(self.prev_waypoint_position, current_position)
        if self.env.show_debug_annotations:
            self.env.p.addUserDebugLine(
                (self.prev_waypoint_position[0], self.prev_waypoint_position[1], 0.001),
                (current_position[0], current_position[1], 0.001),
                VectorEnv.DEBUG_LINE_COLORS[self.group_index]
            )
        self.prev_waypoint_position = current_position

    def _update_pose(self):
        position, orientation = self.env.p.getBasePositionAndOrientation(self.id)
        self._position_raw = position
        self._position = (position[0], position[1], 0)  # Use immutable tuples to represent positions
        self._heading = orientation_to_heading(orientation)
        self._last_step_simulation_count = self.env.step_simulation_count

    def _create_multi_body(self):
        base_height = 0.035
        mass = 0.180
        shape_types = [pybullet.GEOM_CYLINDER, pybullet.GEOM_BOX, pybullet.GEOM_BOX]
        radii = [Robot.HALF_WIDTH, None, None]
        half_extents = [
            None,
            (self.BASE_LENGTH / 2, Robot.HALF_WIDTH, base_height / 2),
            (Robot.TOP_LENGTH / 2, Robot.HALF_WIDTH, Robot.HEIGHT / 2),
        ]
        lengths = [Robot.HEIGHT, None, None]
        rgba_colors = [self.COLOR, None, None]  # pybullet seems to ignore all colors after the first
        frame_positions = [
            (Robot.BACKPACK_OFFSET, 0, Robot.HEIGHT / 2),
            (Robot.BACKPACK_OFFSET + self.BASE_LENGTH / 2, 0, base_height / 2),
            (Robot.BACKPACK_OFFSET + Robot.TOP_LENGTH / 2, 0, Robot.HEIGHT / 2),
        ]
        collision_shape_id = self.env.p.createCollisionShapeArray(
            shapeTypes=shape_types, radii=radii, halfExtents=half_extents, lengths=lengths, collisionFramePositions=frame_positions)
        visual_shape_id = self.env.p.createVisualShapeArray(
            shapeTypes=shape_types, radii=radii, halfExtents=half_extents, lengths=lengths, rgbaColors=rgba_colors, visualFramePositions=frame_positions)
        return self.env.p.createMultiBody(mass, collision_shape_id, visual_shape_id)

    @staticmethod
    def get_robot_cls(robot_type):
        if robot_type == 'pushing_robot':
            return PushingRobot
        if robot_type == 'lifting_robot':
            return LiftingRobot
        if robot_type == 'throwing_robot':
            return ThrowingRobot
        if robot_type == 'rescue_robot':
            return RescueRobot
        raise Exception(robot_type)

    @staticmethod
    def get_robot(robot_type, *args, real=False, real_robot_index=None):
        return Robot.get_robot_cls(robot_type)(*args, real=real, real_robot_index=real_robot_index)

class PushingRobot(Robot):
    BASE_LENGTH = Robot.BASE_LENGTH + 0.005  # 5 mm blade
    END_EFFECTOR_LOCATION = Robot.BACKPACK_OFFSET + BASE_LENGTH
    RADIUS = math.sqrt(Robot.HALF_WIDTH**2 + END_EFFECTOR_LOCATION**2)
    COLOR = (0.1765, 0.1765, 0.1765, 1)  # Dark gray

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cube_dist_closer = 0

    def store_new_action(self, action):
        super().store_new_action(action)
        self.cube_dist_closer = 0

    def process_cube_success(self):
        super().process_cube_success()
        self.cubes_with_reward += 1

    def compute_rewards_and_stats(self, done=False):
        super().compute_rewards_and_stats(done=done)
        partial_rewards = self.env.partial_rewards_scale * self.cube_dist_closer
        self.reward += partial_rewards
        self.cumulative_reward += partial_rewards

    def process_cube_position(self, cube_id, initial_cube_positions):
        if cube_id not in initial_cube_positions:
            return
        cube_position = self.env.get_cube_position(cube_id)
        dist_closer = self.mapper.distance_to_receptacle(initial_cube_positions[cube_id]) - self.mapper.distance_to_receptacle(cube_position)
        self.cube_dist_closer += dist_closer

class RobotWithHooks(Robot):
    NUM_OUTPUT_CHANNELS = 2
    END_EFFECTOR_DIST_THRESHOLD = VectorEnv.CUBE_WIDTH
    END_EFFECTOR_THICKNESS = 0.008  # 8 mm
    END_EFFECTOR_GAP_SIZE = 0.001  # 1 mm gap makes p.stepSimulation faster by avoiding unnecessary collisions

    @abstractmethod  # Should not be instantiated directly
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.potential_cube_id = None
        self.end_effector_id = self._create_end_effector_shape()  # For collision detection (real robot detaches shape while aligning with cube)
        self.end_effector_cid = None
        self.attach_end_effector_shape()

    def reset_pose(self, *args):
        super().reset_pose(*args)
        if self.end_effector_cid is not None:
            # Reset pose of end effector shape
            self.detach_end_effector_shape()
            self.attach_end_effector_shape()

    def ray_test_cube(self):
        # World coordinates
        target_position, target_heading = self.waypoint_positions[-1], self.waypoint_headings[-1]
        ray_from = (
            target_position[0] + math.cos(target_heading) * self.END_EFFECTOR_LOCATION,
            target_position[1] + math.sin(target_heading) * self.END_EFFECTOR_LOCATION,
            VectorEnv.CUBE_WIDTH / 2
        )
        ray_to = (
            target_position[0] + math.cos(target_heading) * self.END_EFFECTOR_LOCATION + math.cos(target_heading) * RobotWithHooks.END_EFFECTOR_DIST_THRESHOLD,
            target_position[1] + math.sin(target_heading) * self.END_EFFECTOR_LOCATION + math.sin(target_heading) * RobotWithHooks.END_EFFECTOR_DIST_THRESHOLD,
            VectorEnv.CUBE_WIDTH / 2
        )
        #self.env.p.addUserDebugLine(ray_from, ray_to, (0, 0, 1))
        body_id = self.env.p.rayTestBatch([ray_from], [ray_to])[0][0]
        if body_id in self.env.available_cube_ids_set:
            return body_id
        return None

    def attach_end_effector_shape(self):
        if self.end_effector_cid is not None:
            return

        # Move to front of robot
        box_width = RobotWithHooks.END_EFFECTOR_THICKNESS - RobotWithHooks.END_EFFECTOR_GAP_SIZE
        x_offset = Robot.BACKPACK_OFFSET + self.BASE_LENGTH + RobotWithHooks.END_EFFECTOR_GAP_SIZE + box_width / 2
        height = RobotWithHooks.END_EFFECTOR_GAP_SIZE + box_width / 2
        current_position, current_heading = self.get_position(), self.get_heading()
        parent_frame_position_world = (
            current_position[0] + x_offset * math.cos(current_heading),
            current_position[1] + x_offset * math.sin(current_heading),
            height
        )
        self.env.p.resetBasePositionAndOrientation(self.end_effector_id, parent_frame_position_world, heading_to_orientation(current_heading))

        # Create constraint
        parent_frame_position = (x_offset, 0, height)
        self.end_effector_cid = self.env.p.createConstraint(self.id, -1, self.end_effector_id, -1, pybullet.JOINT_FIXED, None, parent_frame_position, (0, 0, 0))

        # Collision detection
        self.collision_body_a_ids_set.add(self.end_effector_id)
        self.env.robot_collision_body_b_ids_set.add(self.end_effector_id)

    def detach_end_effector_shape(self):
        self.env.p.removeConstraint(self.end_effector_cid)
        self.end_effector_cid = None
        self.env.p.resetBasePositionAndOrientation(self.end_effector_id, (0, 0, VectorEnv.REMOVED_BODY_Z), VectorEnv.IDENTITY_QUATERNION)
        self.collision_body_a_ids_set.remove(self.end_effector_id)
        self.env.robot_collision_body_b_ids_set.remove(self.end_effector_id)

    def _create_end_effector_shape(self):
        # Add a box to approximate the triangle in front of the robot
        box_width = RobotWithHooks.END_EFFECTOR_THICKNESS - RobotWithHooks.END_EFFECTOR_GAP_SIZE
        half_extents = (box_width / 2, 0.018 / 2, box_width / 2)
        collision_shape_id = self.env.p.createCollisionShape(pybullet.GEOM_BOX, halfExtents=half_extents)
        visual_shape_id = self.env.p.createVisualShape(pybullet.GEOM_BOX, halfExtents=half_extents, rgbaColor=(0, 0, 0, 0))  # Make invisible by setting alpha to 0
        return self.env.p.createMultiBody(0.001, collision_shape_id, visual_shape_id)

class LiftingRobot(RobotWithHooks):
    LIFTED_CUBE_HEIGHT = 0.04
    LIFTED_CUBE_OFFSET = -0.007  # Cube moves backwards towards robot when lifted

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lift_state = 'ready'
        self.lift_cid = None
        self.cube_id = None

        # Partial rewards
        self.initial_cube_position = None
        self.cube_dist_closer = 0

        # Stats
        self.pointless_cube_drop = False

    def store_new_action(self, action):
        super().store_new_action(action)
        self.potential_cube_id = self.ray_test_cube() if (self.lift_state == 'ready' and self.action[0] == 1) else None
        self.cube_dist_closer = 0
        self.pointless_cube_drop = False

    def process_cube_success(self, lifted=False):  # pylint: disable=arguments-differ
        super().process_cube_success()
        if lifted:
            self.cubes_with_reward += 1

    def compute_rewards_and_stats(self, done=False):
        super().compute_rewards_and_stats(done=done)
        partial_rewards = self.env.partial_rewards_scale * self.cube_dist_closer
        self.reward += partial_rewards
        self.cumulative_reward += partial_rewards
        if self.pointless_cube_drop:
            self.reward -= self.env.lifting_pointless_drop_penalty
            self.cumulative_reward -= self.env.lifting_pointless_drop_penalty

    def reset_pose(self, *args):
        super().reset_pose(*args)
        if self.lift_state == 'lifting':
            self._reset_lifted_cube_pose()

    def process_lifted_cube_position(self, cube_position=None):
        if cube_position is None:
            cube_position = self.env.get_cube_position(self.cube_id)
        dist_closer = self.mapper.distance_to_receptacle(self.initial_cube_position) - self.mapper.distance_to_receptacle(cube_position)
        self.cube_dist_closer += dist_closer
        self.initial_cube_position = self.env.get_cube_position(self.cube_id)

    def lift_cube(self, cube_id):
        # Update variables and environment state
        self.cube_id = cube_id
        self.lift_state = 'lifting'
        self.env.available_cube_ids_set.remove(self.cube_id)
        self.collision_body_a_ids_set.add(self.cube_id)
        self.env.robot_collision_body_b_ids_set.add(self.cube_id)

        # Store initial cube position for partial rewards
        self.initial_cube_position = self.env.get_cube_position(self.cube_id)

        # Move cube to lifted position
        self._reset_lifted_cube_pose()

        # Create constraint
        offset = LiftingRobot.END_EFFECTOR_LOCATION + LiftingRobot.LIFTED_CUBE_OFFSET + VectorEnv.CUBE_WIDTH / 2
        parent_frame_position = (offset, 0, LiftingRobot.LIFTED_CUBE_HEIGHT + VectorEnv.CUBE_WIDTH / 2)
        self.lift_cid = self.env.p.createConstraint(self.id, -1, self.cube_id, -1, pybullet.JOINT_FIXED, None, parent_frame_position, (0, 0, 0))

    def drop_cube(self):
        # Process final cube position for partial rewards
        cube_position = self.env.get_cube_position(self.cube_id)
        self.process_lifted_cube_position(cube_position)

        # Remove constraint
        self.env.p.removeConstraint(self.lift_cid)
        self.lift_cid = None

        # Move cube forward beyond end effector
        current_position, current_heading = self.get_position(), self.get_heading()
        offset = LiftingRobot.END_EFFECTOR_LOCATION + RobotWithHooks.END_EFFECTOR_THICKNESS + VectorEnv.CUBE_WIDTH / 2
        cube_position = (
            current_position[0] + offset * math.cos(current_heading),
            current_position[1] + offset * math.sin(current_heading),
            Robot.HEIGHT + VectorEnv.CUBE_WIDTH / 2
        )
        self.env.p.resetBasePositionAndOrientation(self.cube_id, cube_position, heading_to_orientation(current_heading))

        # Update variables and environment state
        if self.env.cube_position_in_receptacle(cube_position):
            self.process_cube_success(lifted=True)
            self.env.remove_cube(self.cube_id)
        else:
            self.env.available_cube_ids_set.add(self.cube_id)
            self.pointless_cube_drop = True
        self.lift_state = 'ready'
        self.collision_body_a_ids_set.remove(self.cube_id)
        self.env.robot_collision_body_b_ids_set.remove(self.cube_id)
        self.initial_cube_position = None
        self.cube_id = None

    def _reset_lifted_cube_pose(self):
        current_position, current_heading = self.get_position(), self.get_heading()
        offset = LiftingRobot.END_EFFECTOR_LOCATION + LiftingRobot.LIFTED_CUBE_OFFSET + VectorEnv.CUBE_WIDTH / 2
        cube_position = (
            current_position[0] + offset * math.cos(current_heading),
            current_position[1] + offset * math.sin(current_heading),
            LiftingRobot.LIFTED_CUBE_HEIGHT + VectorEnv.CUBE_WIDTH / 2
        )
        self.env.p.resetBasePositionAndOrientation(self.cube_id, cube_position, heading_to_orientation(current_heading))

class ThrowingRobot(RobotWithHooks):
    BASE_LENGTH = Robot.BASE_LENGTH + 0.006  # 6 mm offset
    END_EFFECTOR_LOCATION = Robot.BACKPACK_OFFSET + BASE_LENGTH
    RADIUS = math.sqrt(Robot.HALF_WIDTH**2 + END_EFFECTOR_LOCATION**2)
    COLOR = (0.5294, 0.5294, 0.5294, 1)  # Light gray

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cube_id = None
        self.initial_cube_position = None
        self.cube_dist_closer = 0

    def store_new_action(self, action):
        super().store_new_action(action)
        self.potential_cube_id = self.ray_test_cube() if self.action[0] == 1 else None
        self.cube_dist_closer = 0

    def process_cube_success(self, thrown=False):  # pylint: disable=arguments-differ
        super().process_cube_success()
        if thrown:
            self.cubes_with_reward += 1

    def compute_rewards_and_stats(self, done=False):
        super().compute_rewards_and_stats(done=done)
        partial_rewards = self.env.partial_rewards_scale * self.cube_dist_closer
        self.reward += partial_rewards
        self.cumulative_reward += partial_rewards

    def prepare_throw_cube(self, cube_id):
        # Update variables and environment state
        self.cube_id = cube_id
        self.env.available_cube_ids_set.remove(self.cube_id)

        # Store initial cube position for partial rewards
        self.initial_cube_position = self.env.get_cube_position(self.cube_id)

    def throw_cube(self):
        assert not self.real
        # Move cube over back of robot
        current_position, current_heading = self.get_position(), self.get_heading()
        back_position = (
            current_position[0] + Robot.BACKPACK_OFFSET * math.cos(current_heading),
            current_position[1] + Robot.BACKPACK_OFFSET * math.sin(current_heading),
            Robot.HEIGHT + VectorEnv.CUBE_WIDTH  # Half cube width above robot
        )
        self.env.p.resetBasePositionAndOrientation(self.cube_id, back_position, heading_to_orientation(current_heading))

        # Apply force and torque
        force_x = self.env.robot_random_state.normal(5.5, 0.75)
        force_y = self.env.robot_random_state.normal(1.5, 0.75) * (-1 if self.env.robot_random_state.rand() < 0.5 else 1)
        self.env.p.applyExternalForce(self.cube_id, -1, (-force_x, -force_y, 0), (0, 0, 0), flags=pybullet.LINK_FRAME)
        self.env.p.applyExternalTorque(self.cube_id, -1, (0, -0.03, 0), flags=pybullet.WORLD_FRAME)  # See https://github.com/bulletphysics/bullet3/issues/1949

    def finish_throw_cube(self):
        # Process final cube position for partial rewards
        cube_position = self.env.get_cube_position(self.cube_id)
        dist_closer = self.mapper.distance_to_receptacle(self.initial_cube_position) - self.mapper.distance_to_receptacle(cube_position)
        self.cube_dist_closer += dist_closer

        # Update variables and environment state
        if self.env.cube_position_in_receptacle(cube_position):
            self.process_cube_success(thrown=True)
            self.env.remove_cube(self.cube_id)
        else:
            self.env.available_cube_ids_set.add(self.cube_id)
        self.cube_id = None

class RescueRobot(RobotWithHooks):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cube_id = None

    def store_new_action(self, action):
        super().store_new_action(action)
        self.potential_cube_id = self.ray_test_cube() if self.action[0] == 1 else None

    def process_cube_success(self):
        super().process_cube_success()
        self.cubes_with_reward += 1

    def prepare_rescue_cube(self, cube_id):
        self.cube_id = cube_id
        self.env.available_cube_ids_set.remove(cube_id)

    def rescue_cube(self):
        # Update variables and environment state
        self.process_cube_success()
        self.env.remove_cube(self.cube_id)
        self.cube_id = None

class RobotController:
    DRIVE_STEP_SIZE = 0.005  # 5 mm results in exactly 1 mm per simulation step
    TURN_STEP_SIZE = math.radians(15)  # 15 deg results in exactly 3 deg per simulation step

    def __init__(self, robot):
        self.robot = robot
        self.state = 'idle'
        self.waypoint_index = None  # Index of waypoint we are currently headed towards
        self.prev_position = None  # Position before call to p.stepSimulation()
        self.prev_heading = None
        self.sim_steps = 0
        self.consecutive_turning_sim_steps = None  # Used to detect if robot is stuck and oscillating
        self.manipulation_sim_step_target = 0
        self.manipulation_sim_steps = 0

    def reset(self):
        self.state = 'idle'
        self.waypoint_index = 1
        self.prev_position = None
        self.prev_heading = None
        self.sim_steps = 0
        self.consecutive_turning_sim_steps = 0

    def new_action(self):
        self.state = 'moving'

    def step(self):
        # States: idle, moving, manipulating

        assert not self.state == 'idle'
        self.sim_steps += 1

        # Periodically update the map
        if self.sim_steps % 200 == 0:
            self.robot.update_map()

        if self.state == 'moving':
            current_position, current_heading = self.robot.get_position(), self.robot.get_heading()

            # First check change after sim step
            if self.prev_position is not None:

                # Detect if robot is still moving
                driving = distance(self.prev_position, current_position) > 0.0005  # 0.5 mm
                turning = abs(heading_difference(self.prev_heading, current_heading)) > math.radians(1)  # 1 deg
                self.consecutive_turning_sim_steps = (self.consecutive_turning_sim_steps + 1) if turning else 0
                stuck_oscillating = self.consecutive_turning_sim_steps > 100  # About 60 sim steps is sufficient for turning 180 deg
                not_moving = (not driving and not turning) or stuck_oscillating

                # Check for collisions
                if distance(self.robot.waypoint_positions[0], current_position) > RobotController.DRIVE_STEP_SIZE or not_moving:
                    self.robot.check_for_collisions()

                # Check if step limit exceeded (expect this won't ever happen, but just in case)
                step_limit_exceeded = self.sim_steps > 3200

                if self.robot.collided_with_obstacle or self.robot.collided_with_robot or step_limit_exceeded:
                    self.robot.update_distance()
                    self.state = 'idle'

                if self.state == 'moving' and not_moving:
                    # Reached current waypoint, move on to next waypoint
                    self.robot.update_distance()
                    if self.waypoint_index == len(self.robot.waypoint_positions) - 1:
                        self._done_moving()
                    else:
                        self.waypoint_index += 1

            # If still moving, set constraint for new pose
            if self.state == 'moving':
                new_position, new_heading = current_position, current_heading

                # Determine whether to turn or drive
                heading_diff = heading_difference(current_heading, self.robot.waypoint_headings[self.waypoint_index])
                if abs(heading_diff) > RobotController.TURN_STEP_SIZE:
                    new_heading += math.copysign(1, heading_diff) * RobotController.TURN_STEP_SIZE
                else:
                    curr_waypoint_position = self.robot.waypoint_positions[self.waypoint_index]
                    dx = curr_waypoint_position[0] - current_position[0]
                    dy = curr_waypoint_position[1] - current_position[1]
                    if distance(current_position, curr_waypoint_position) < RobotController.DRIVE_STEP_SIZE:
                        new_position = curr_waypoint_position
                    else:
                        move_sign = math.copysign(1, distance(current_position, self.robot.target_end_effector_position) - (self.robot.END_EFFECTOR_LOCATION + VectorEnv.CUBE_WIDTH / 2))
                        new_heading = math.atan2(move_sign * dy, move_sign * dx)
                        new_position = (
                            new_position[0] + move_sign * RobotController.DRIVE_STEP_SIZE * math.cos(new_heading),
                            new_position[1] + move_sign * RobotController.DRIVE_STEP_SIZE * math.sin(new_heading),
                            new_position[2]
                        )

                # Set constraint
                self.robot.env.p.changeConstraint(
                    self.robot.cid, jointChildPivot=new_position, jointChildFrameOrientation=heading_to_orientation(new_heading), maxForce=Robot.CONSTRAINT_MAX_FORCE)

            self.prev_position, self.prev_heading = current_position, current_heading

        elif self.state == 'manipulating':
            self.manipulation_sim_steps += 1
            if self.manipulation_sim_steps >= self.manipulation_sim_step_target:
                self.manipulation_sim_step_target = 0
                self.manipulation_sim_steps = 0
                if isinstance(self.robot, ThrowingRobot):
                    self.robot.finish_throw_cube()
                self.state = 'idle'

    def get_intention_path(self):
        return [self.robot.get_position()] + self.robot.waypoint_positions[self.waypoint_index:-1] + [self.robot.target_end_effector_position]

    def get_history_path(self):
        return self.robot.waypoint_positions[:self.waypoint_index] + [self.robot.get_position()]

    def _done_moving(self):
        self.state = 'idle'
        if isinstance(self.robot, LiftingRobot) and self.robot.lift_state == 'lifting':
            if self.robot.action[0] == 1:
                self.robot.drop_cube()
                self.state = 'manipulating'
                self.manipulation_sim_step_target = 30
            else:
                self.robot.process_lifted_cube_position()
        elif isinstance(self.robot, RobotWithHooks):
            if self.robot.potential_cube_id is not None and distance(self.robot.get_position(), self.robot.waypoint_positions[-1]) < RobotController.DRIVE_STEP_SIZE:
                cube_id = self.robot.ray_test_cube()
                if cube_id is not None:
                    if isinstance(self.robot, LiftingRobot):
                        self.robot.lift_cube(cube_id)
                    elif isinstance(self.robot, ThrowingRobot):
                        self.robot.prepare_throw_cube(cube_id)
                        self.robot.throw_cube()
                        self.state = 'manipulating'
                        self.manipulation_sim_step_target = 100
                    elif isinstance(self.robot, RescueRobot):
                        self.robot.prepare_rescue_cube(cube_id)
                        self.robot.rescue_cube()

class RealRobotController:
    LOOKAHEAD_DISTANCE = 0.1  # 10 cm
    TURN_THRESHOLD = math.radians(5)  # 5 deg

    def __init__(self, robot, real_robot_index, debug=False):
        self.robot = robot
        self.real_robot_name = vector_utils.get_robot_name(real_robot_index)
        self.debug = debug
        self.real_robot = anki_vector.AsyncRobot(serial=vector_utils.get_robot_serial(real_robot_index), default_logging=False, behavior_control_level=anki_vector.connection.ControlPriorityLevel.OVERRIDE_BEHAVIORS_PRIORITY)
        self.real_robot.connect()
        battery_state = self.real_robot.get_battery_state().result()
        battery_volts = '{:.2f}'.format(battery_state.battery_volts) if battery_state else '?'
        print('Connected to {} ({} V)'.format(self.real_robot_name, battery_volts))
        self._reset_motors()

        self.state = 'idle'
        self.resume_state = None  # For pausing
        self.waypoint_index = None  # Index of waypoint we are currently headed towards
        self.prev_position = None
        self.prev_heading = None
        self.sim_steps = 0
        self.target_cube_id = None
        self.not_driving_sim_steps = None
        self.not_turning_sim_steps = None
        self.cube_sim_steps = None  # For monitoring lifted, thrown, or rescued cubes
        self.lifting_sim_steps = None  # For handling failed lifts
        self.throwing_sim_steps = None  # For handling failed throws

        if self.debug:
            self.debug_data = None

    def reset(self):
        self.real_robot.motors.set_wheel_motors(0, 0)

        if not isinstance(self.robot, LiftingRobot):
            self.real_robot.behavior.set_lift_height(0)

        if isinstance(self.robot, RobotWithHooks):
            self.robot.attach_end_effector_shape()

        self.state = 'idle'
        self.resume_state = None
        self.waypoint_index = 1
        self.prev_position = None
        self.prev_heading = None
        self.sim_steps = 0
        self.target_cube_id = None
        self.not_driving_sim_steps = 0
        self.not_turning_sim_steps = 0
        self.cube_sim_steps = 0
        self.lifting_sim_steps = 0
        self.throwing_sim_steps = 0

        if self.debug:
            self.debug_data = None

    def new_action(self):
        self.state = 'turning'

    def step(self):
        # States: idle, stopping, turning, driving, slowing, aligning, lifting, rescuing, throwing, pulling

        if self.state == 'idle':
            return

        self.sim_steps += 1

        # Periodically update the map (map updates are slow)
        if self.sim_steps % 20 == 0:
            self.robot.update_map()

        # Update target end effector position to track target cube
        if self.state in {'aligning', 'pulling'}:
            # Sometimes there is another cube between the robot and the target cube
            cube_id = self.robot.ray_test_cube()
            if cube_id is not None:
                self.target_cube_id = cube_id
            self.robot.target_end_effector_position = self.robot.env.get_cube_position(self.target_cube_id)

        if self.state == 'stopping':
            self.real_robot.motors.set_wheel_motors(0, 0)
            if not self.real_robot.status.are_wheels_moving:
                self._done_stopping()

        elif self.state in {'turning', 'driving', 'slowing', 'aligning'}:
            current_position, current_heading = self.robot.get_position(), self.robot.get_heading()

            lookahead_position = self._get_lookahead_position()
            dx = lookahead_position[0] - current_position[0]
            dy = lookahead_position[1] - current_position[1]
            heading_diff = heading_difference(current_heading, math.atan2(dy, dx))

            if self.debug:
                self.debug_data = (lookahead_position, None, None, None, None)

            if self.state == 'turning':
                if abs(heading_diff) < RealRobotController.TURN_THRESHOLD:
                    self.real_robot.motors.set_wheel_motors(0, 0)
                    if not self.real_robot.status.are_wheels_moving:
                        self.state = 'driving'
                else:
                    speed = max(20, min(100, 100 * abs(heading_diff)))  # Must be at least 20 for marker detection to detect changes

                    if self.prev_heading is not None:
                        # Detect if robot is turning more slowly than expected
                        if abs(heading_difference(self.prev_heading, current_heading)) < speed / 2000:
                            self.not_turning_sim_steps += 1
                        else:
                            self.not_turning_sim_steps = 0
                        #print(self.not_turning_sim_steps, abs(heading_difference(self.prev_heading, current_heading)), speed / 2000)
                        if self.not_turning_sim_steps > 20:
                            self.real_robot.motors.set_wheel_motors(0, 0)
                            self.state = 'stopping'

                    if self.state == 'turning':
                        sign = math.copysign(1, heading_diff)
                        self.real_robot.motors.set_wheel_motors(-1 * sign * speed, sign * speed)

            elif self.state in {'driving', 'slowing', 'aligning'}:
                signed_dist = distance(current_position, self.robot.target_end_effector_position) - (self.robot.END_EFFECTOR_LOCATION + VectorEnv.CUBE_WIDTH / 2)
                speed = max(20, min(100, 2000 * abs(signed_dist))) if self.state == 'slowing' else 100  # Must be at least 20 for marker detection to detect changes

                if self.prev_position is not None:
                    # Detect if robot is driving more slowly than expected
                    if distance(self.prev_position, current_position) < speed / 40000:
                        self.not_driving_sim_steps += 1
                    else:
                        self.not_driving_sim_steps = 0
                    #print(self.not_driving_sim_steps, distance(self.prev_position, current_position), speed / 40000)

                    # Check for collisions (It would be nice to have collision detection while turning too, but that is not currently implemented)
                    if distance(self.robot.waypoint_positions[0], current_position) > 0.01 or self.not_driving_sim_steps > 20:
                        self.robot.check_for_collisions()

                if self.robot.collided_with_obstacle or self.robot.collided_with_robot or self.not_driving_sim_steps > 20:
                    self.real_robot.motors.set_wheel_motors(0, 0)
                    self.state = 'stopping'

                elif self.state == 'driving' and signed_dist < VectorEnv.CUBE_WIDTH:
                    self._done_driving()

                elif self.state == 'slowing' and abs(signed_dist) < 0.002:  # 2 mm
                    self._done_slowing()

                elif self.state == 'aligning' and abs(heading_diff) < RealRobotController.TURN_THRESHOLD and signed_dist < 0.001:  # 1 mm buffer for the hooks
                    # If marker detection fails to detect the target cube, we might get a false positive here since the cube pose will be outdated
                    self._done_aligning()

                else:
                    # Pure pursuit
                    lookahead_dist = math.sqrt(dx**2 + dy**2)
                    signed_radius = lookahead_dist / (2 * math.sin(heading_diff))
                    sign = math.copysign(1, signed_dist)
                    wheel_width = 0.1  # 10 cm (larger than actual width due to tread slip)
                    left_wheel_speed = sign * speed * (signed_radius - sign * wheel_width / 2) / signed_radius
                    right_wheel_speed = sign * speed * (signed_radius + sign * wheel_width / 2) / signed_radius

                    # Turn more forcefully if stuck
                    if isinstance(self.robot, PushingRobot) and abs(heading_diff) > RealRobotController.TURN_THRESHOLD and self.not_driving_sim_steps > 10:
                        if left_wheel_speed > right_wheel_speed:
                            right_wheel_speed = -left_wheel_speed
                        else:
                            left_wheel_speed = -right_wheel_speed

                    self.real_robot.motors.set_wheel_motors(left_wheel_speed, right_wheel_speed)

                    if self.debug:
                        self.debug_data = (lookahead_position, signed_radius, heading_diff, current_position, current_heading)

            self.prev_position, self.prev_heading = current_position, current_heading

        elif self.state == 'lifting':
            self.lifting_sim_steps += 1
            if self.lifting_sim_steps > 20:  # Cube is probably stuck against a wall
                self.lifting_sim_steps = 0
                self.state = 'stopping'
            elif self.real_robot.lift_height_mm > self._lift_height_to_mm(0.5):
                self.lifting_sim_steps = 0
                self._done_lifting()

        elif self.state == 'throwing':
            if self.real_robot.lift_height_mm > self._lift_height_to_mm(0.5):
                self.cube_sim_steps += 1
                if self.cube_sim_steps > 20:
                    self.cube_sim_steps = 0
                    self._done_throwing()
            else:
                self.throwing_sim_steps += 1
                if self.throwing_sim_steps > 10:
                    self.throwing_sim_steps = 0
                    self._failed_throwing()

        elif self.state == 'pulling':
            self.throwing_sim_steps += 1
            if self.throwing_sim_steps > 10:
                self.throwing_sim_steps = 0
                self._done_pulling()

        elif self.state == 'rescuing':
            if self.cube_sim_steps > 10:
                self.cube_sim_steps = 0
                self._done_rescuing()

    def get_intention_path(self):
        if self.state == 'idle':
            return None
        lookahead_position = self._get_lookahead_position()
        intermediate_waypoint = (lookahead_position[0], lookahead_position[1], 0)
        return [self.robot.get_position(), intermediate_waypoint] + self.robot.waypoint_positions[self.waypoint_index:-1] + [self.robot.target_end_effector_position]

    def get_history_path(self):
        if self.state == 'idle':
            return None
        current_position = self.robot.get_position()
        closest_waypoint_index = self.waypoint_index
        while closest_waypoint_index > 0:
            start = self.robot.waypoint_positions[closest_waypoint_index - 1]
            end = self.robot.waypoint_positions[closest_waypoint_index]
            d = (end[0] - start[0], end[1] - start[1])
            f = (start[0] - current_position[0], start[1] - current_position[1])
            t1 = self._intersect(d, f, RealRobotController.LOOKAHEAD_DISTANCE, use_t1=True)
            if t1 is not None:
                intermediate_waypoint = (start[0] + t1 * d[0], start[1] + t1 * d[1], 0)
                return self.robot.waypoint_positions[:closest_waypoint_index] + [intermediate_waypoint, current_position]
            closest_waypoint_index -= 1
        return [self.robot.waypoint_positions[0], current_position]

    def pause(self):
        if self.state != 'idle':
            self.resume_state = self.state
            self.real_robot.motors.set_wheel_motors(0, 0)
            self.state = 'stopping'

    def resume(self):
        if self.resume_state is not None:
            self.state = self.resume_state
            self.resume_state = None

    def disconnect(self):
        self._reset_motors()
        self.real_robot.disconnect()
        print('Disconnected from {}'.format(self.real_robot_name))

    def monitor_lifted_cube(self, estimated_cube_pose):
        # Note: Since the lifted cube does not lie in the same plane, the cube position is only a rough estimate
        if distance(self.robot.get_position(), estimated_cube_pose['position']) > 0.1:  # 10 cm
            self.cube_sim_steps += 1
        else:
            self.cube_sim_steps = 0

        if self.cube_sim_steps > 10:
            self.real_robot.behavior.set_lift_height(0)
            self.robot.drop_cube()

    def monitor_rescued_cube(self, estimated_cube_pose):
        if estimated_cube_pose is None:
            self.cube_sim_steps += 1
        else:
            self.cube_sim_steps = 0

    def _done_stopping(self):
        self.robot.update_distance()
        self.state = 'paused' if self.resume_state is not None else 'idle'

    def _done_driving(self):
        self.state = 'slowing'
        if isinstance(self.robot, RobotWithHooks) and self.robot.potential_cube_id is not None:
            cube_id = self.robot.ray_test_cube()
            if cube_id is not None:
                self.robot.update_distance()
                self.robot.detach_end_effector_shape()
                self.target_cube_id = cube_id
                self.state = 'aligning'

    def _done_slowing(self):
        self.real_robot.motors.set_wheel_motors(0, 0)
        self.state = 'stopping'

        if isinstance(self.robot, LiftingRobot) and self.robot.lift_state == 'lifting':
            if self.robot.action[0] == 1:
                # Drop cube
                self.real_robot.behavior.set_lift_height(0)
                self.robot.drop_cube()
                for _ in range(30):  # Wait for simulation to update
                    self.robot.env.step_simulation()
            else:
                # Give lifting partial rewards
                self.robot.process_lifted_cube_position()

    def _done_aligning(self):
        self.real_robot.motors.set_wheel_motors(0, 0)
        if isinstance(self.robot, (LiftingRobot, RescueRobot)):
            self.real_robot.behavior.set_lift_height(0.85)  # Using 1.0 causes the marker on top of the robot to be occluded
            self.state = 'lifting'
        elif isinstance(self.robot, ThrowingRobot):
            self.robot.prepare_throw_cube(self.target_cube_id)
            self.real_robot.motors.set_lift_motor(8.0)
            self.state = 'throwing'

    def _done_lifting(self):
        if isinstance(self.robot, LiftingRobot):
            self.robot.lift_cube(self.target_cube_id)
            self.state = 'stopping'
        elif isinstance(self.robot, RescueRobot):
            self.robot.prepare_rescue_cube(self.target_cube_id)
            self.cube_sim_steps = 0
            self.state = 'rescuing'

    def _done_throwing(self):
        self.real_robot.motors.set_lift_motor(0)
        self.real_robot.behavior.set_lift_height(0)
        self.robot.finish_throw_cube()
        self.state = 'stopping'

    def _failed_throwing(self):
        self.real_robot.motors.set_lift_motor(0)
        self.real_robot.motors.set_wheel_motors(-40, -40)
        self.state = 'pulling'

    def _done_pulling(self):
        self.real_robot.motors.set_wheel_motors(0, 0)
        self.real_robot.motors.set_lift_motor(8.0)
        self.state = 'throwing'

    def _done_rescuing(self):
        self.real_robot.behavior.set_lift_height(0)
        self.robot.rescue_cube()
        self.state = 'stopping'

    def _reset_motors(self):
        self.real_robot.motors.set_wheel_motors(0, 0)
        self.real_robot.behavior.set_lift_height(0)
        self.real_robot.behavior.set_head_angle(anki_vector.util.degrees(0))

    def _get_lookahead_position(self):
        current_position = self.robot.get_position()
        while True:
            start = self.robot.waypoint_positions[self.waypoint_index - 1]
            end = self.robot.waypoint_positions[self.waypoint_index]
            d = (end[0] - start[0], end[1] - start[1])
            f = (start[0] - current_position[0], start[1] - current_position[1])
            t2 = self._intersect(d, f, RealRobotController.LOOKAHEAD_DISTANCE)
            if t2 is not None:
                return (start[0] + t2 * d[0], start[1] + t2 * d[1])
            if self.waypoint_index == len(self.robot.waypoint_positions) - 1:
                return self.robot.target_end_effector_position
            self.robot.update_distance()
            self.waypoint_index += 1

    @staticmethod
    def _intersect(d, f, r, use_t1=False):
        # https://stackoverflow.com/questions/1073336/circle-line-segment-collision-detection-algorithm/1084899%231084899
        a = dot(d, d)
        b = 2 * dot(f, d)
        c = dot(f, f) - r * r
        discriminant = (b * b) - (4 * a * c)
        if discriminant >= 0:
            if use_t1:
                t1 = (-b - math.sqrt(discriminant)) / (2 * a + 1e-6)
                if 0 <= t1 <= 1:
                    return t1
            else:
                t2 = (-b + math.sqrt(discriminant)) / (2 * a + 1e-6)
                if 0 <= t2 <= 1:
                    return t2
        return None

    @staticmethod
    def _lift_height_to_mm(height):
        return anki_vector.behavior.MIN_LIFT_HEIGHT_MM + height * (anki_vector.behavior.MAX_LIFT_HEIGHT_MM - anki_vector.behavior.MIN_LIFT_HEIGHT_MM)

class Camera(ABC):
    NEAR = None
    FAR = None
    ASPECT = None
    FOV = 60  # Vertical FOV
    SEG_VALUES = {
        'floor': 1.0 / 8,
        'obstacle': 2.0 / 8,
        'receptacle': 3.0 / 8,
        'cube': 4.0 / 8,
        'robot_group_1': 5.0 / 8,
        'robot_group_2': 6.0 / 8,
        'robot_group_3': 7.0 / 8,
        'robot_group_4': 8.0 / 8,
    }

    @abstractmethod  # Should not be instantiated directly
    def __init__(self, env):
        self.env = env
        self.image_pixel_height = int(1.63 * Mapper.LOCAL_MAP_PIXEL_WIDTH)
        self.image_pixel_width = int(self.ASPECT * self.image_pixel_height)
        self.projection_matrix = self.env.p.computeProjectionMatrixFOV(Camera.FOV, self.ASPECT, self.NEAR, self.FAR)
        self._initialized = False

        # Body ids for constructing the segmentation
        self.min_obstacle_id = None
        self.max_obstacle_id = None
        self.receptacle_id = None
        self.min_cube_id = None
        self.max_cube_id = None

    def _ensure_initialized(self):
        if self._initialized:
            return

        # Note: This should be called after the environment is fully created
        self.min_obstacle_id = min(self.env.obstacle_ids)
        self.max_obstacle_id = max(self.env.obstacle_ids)
        self.receptacle_id = self.env.receptacle_id
        self.min_cube_id = min(self.env.cube_ids)
        self.max_cube_id = max(self.env.cube_ids)
        self._initialized = True

    def capture_image(self, robot_position, robot_heading):
        self._ensure_initialized()

        # Capture images
        camera_position, camera_target, camera_up = self._get_camera_params(robot_position, robot_heading)
        view_matrix = self.env.p.computeViewMatrix(camera_position, camera_target, camera_up)
        images = self.env.p.getCameraImage(self.image_pixel_width, self.image_pixel_height, view_matrix, self.projection_matrix)

        # Compute depth
        depth_buffer = np.reshape(images[3], (self.image_pixel_height, self.image_pixel_width))
        depth = self.FAR * self.NEAR / (self.FAR - (self.FAR - self.NEAR) * depth_buffer)

        # Construct point cloud
        camera_position = np.array(camera_position, dtype=np.float32)
        principal = np.array(camera_target, dtype=np.float32) - camera_position
        principal = principal / np.linalg.norm(principal)
        camera_up = np.array(camera_up, dtype=np.float32)
        up = camera_up - np.dot(camera_up, principal) * principal
        up = up / np.linalg.norm(up)
        right = np.cross(principal, up)
        right = right / np.linalg.norm(right)
        limit_y = math.tan(math.radians(Camera.FOV / 2))
        limit_x = limit_y * self.ASPECT
        pixel_x = (2 * limit_x) * (np.arange(self.image_pixel_width, dtype=np.float32) / self.image_pixel_width - 0.5)
        pixel_y = (2 * limit_y) * (0.5 - (np.arange(self.image_pixel_height, dtype=np.float32) + 1) / self.image_pixel_height)
        pixel_xv, pixel_yv = np.meshgrid(pixel_x, pixel_y)
        points = camera_position + depth[:, :, np.newaxis] * (principal + pixel_xv[:, :, np.newaxis] * right + pixel_yv[:, :, np.newaxis] * up)

        # Construct segmentation
        seg_raw = np.reshape(images[4], (self.image_pixel_height, self.image_pixel_width))
        seg = Camera.SEG_VALUES['floor'] * (seg_raw == 0).astype(np.float32)
        seg += Camera.SEG_VALUES['obstacle'] * np.logical_and(seg_raw >= self.min_obstacle_id, seg_raw <= self.max_obstacle_id).astype(np.float32)
        if self.receptacle_id is not None:
            seg += Camera.SEG_VALUES['receptacle'] * (seg_raw == self.receptacle_id).astype(np.float32)
        seg += Camera.SEG_VALUES['cube'] * np.logical_and(seg_raw >= self.min_cube_id, seg_raw <= self.max_cube_id).astype(np.float32)

        return points, seg

    def get_seg_value(self, body_type):
        self._ensure_initialized()
        return Camera.SEG_VALUES[body_type]

    @abstractmethod
    def _get_camera_params(self, robot_position, robot_heading):
        pass

class OverheadCamera(Camera):
    HEIGHT = 1  # 1 m
    ASPECT = 1
    NEAR = 0.1  # 10 cm
    FAR = 10  # 10 m

    def __init__(self, env):
        super().__init__(env)

    def _get_camera_params(self, robot_position, robot_heading):
        camera_position = (robot_position[0], robot_position[1], OverheadCamera.HEIGHT)
        camera_target = (robot_position[0], robot_position[1], 0)
        camera_up = (math.cos(robot_heading), math.sin(robot_heading), 0)
        return camera_position, camera_target, camera_up

class ForwardFacingCamera(Camera):
    HEIGHT = Robot.HEIGHT
    PITCH = -30
    ASPECT = 16.0 / 9  # 60 deg vertical FOV, 90 deg horizontal FOV
    NEAR = 0.001  # 1 mm
    FAR = 1  # 1 m

    def __init__(self, env):
        super().__init__(env)

    def _get_camera_params(self, robot_position, robot_heading):
        camera_position_offset = Robot.BACKPACK_OFFSET + Robot.TOP_LENGTH + 0.002  # Move forward additional 2 mm (past lifted cube)
        camera_position = (
            robot_position[0] + camera_position_offset * math.cos(robot_heading),
            robot_position[1] + camera_position_offset * math.sin(robot_heading),
            ForwardFacingCamera.HEIGHT
        )
        camera_target_offset = ForwardFacingCamera.HEIGHT * math.tan(math.radians(90 + ForwardFacingCamera.PITCH))
        camera_target = (
            camera_position[0] + camera_target_offset * math.cos(robot_heading),
            camera_position[1] + camera_target_offset * math.sin(robot_heading),
            0
        )
        camera_up = (
            math.cos(math.radians(90 + ForwardFacingCamera.PITCH)) * math.cos(robot_heading),
            math.cos(math.radians(90 + ForwardFacingCamera.PITCH)) * math.sin(robot_heading),
            math.sin(math.radians(90 + ForwardFacingCamera.PITCH))
        )
        return camera_position, camera_target, camera_up

class Mapper:
    LOCAL_MAP_PIXEL_WIDTH = 96
    LOCAL_MAP_WIDTH = 1  # 1 meter
    LOCAL_MAP_PIXELS_PER_METER = LOCAL_MAP_PIXEL_WIDTH / LOCAL_MAP_WIDTH

    def __init__(self, env, robot):
        self.env = env
        self.robot = robot

        # Camera
        if self.env.use_partial_observations:
            self.camera = ForwardFacingCamera(self.env)
        else:
            self.camera = OverheadCamera(self.env)

        # Overhead map
        self.global_overhead_map_without_robots = self._create_padded_room_zeros()

        # Occupancy map
        self.global_occupancy_map = OccupancyMap(self.robot, self.env.room_length, self.env.room_width, show_map=self.env.show_occupancy_maps)

        # Robot masks for overhead map and robot map
        self.robot_masks = {}
        for g in self.env.robot_config:
            robot_type = next(iter(g))
            robot_cls = Robot.get_robot_cls(robot_type)
            self.robot_masks[robot_cls] = self._create_robot_mask(robot_cls)
            if robot_cls == LiftingRobot:
                self.robot_masks['lifting_robot_with_cube'] = self._create_robot_mask(LiftingRobot, show_lifted_cube=True)

        # Precompute global distance to receptacle map
        if self.env.use_distance_to_receptacle_map:
            self.global_distance_to_receptacle_map = self._create_global_distance_to_receptacle_map()

        # Intention map
        self.intention_map_selem = disk(self.env.intention_map_line_thickness - 1)

        # Assertions
        if any(isinstance(robot, RescueRobot) for robot in self.env.robots):
            assert not self.env.use_distance_to_receptacle_map
            assert not self.env.use_shortest_path_to_receptacle_map
        if self.env.use_distance_to_receptacle_map or self.env.use_shortest_path_to_receptacle_map:
            assert self.env.receptacle_id is not None

    def update(self):
        # Get new observation
        points, seg = self.camera.capture_image(self.robot.get_position(), self.robot.get_heading())
        augmented_points = np.concatenate((points, seg[:, :, np.newaxis]), axis=2).reshape(-1, 4)
        augmented_points = augmented_points[np.argsort(augmented_points[:, 2])]

        # Incorporate new observation into overhead map
        pixel_i, pixel_j = Mapper.position_to_pixel_indices(augmented_points[:, 0], augmented_points[:, 1], self.global_overhead_map_without_robots.shape)
        self.global_overhead_map_without_robots[pixel_i, pixel_j] = augmented_points[:, 3]

        # Update occupancy map
        if self.global_occupancy_map is not None:
            self.global_occupancy_map.update(points, seg, self.camera.get_seg_value('obstacle'))

    def get_state(self, save_figures=False):
        channels = []

        # Overhead map
        global_overhead_map = self._create_global_overhead_map()
        local_overhead_map = self._get_local_map(global_overhead_map)
        channels.append(local_overhead_map)

        # Robot map
        if self.env.use_robot_map:
            global_robot_map = self._create_global_robot_map(seg=False)
            local_robot_map = self._get_local_map(global_robot_map)
            channels.append(local_robot_map)

        # Distance to receptacle map
        if self.env.use_distance_to_receptacle_map:
            channels.append(self._get_local_distance_map(self.global_distance_to_receptacle_map))

        # Shortest path distance to receptacle map
        if self.env.use_shortest_path_to_receptacle_map:
            global_shortest_path_to_receptacle_map = self._create_global_shortest_path_to_receptacle_map()
            local_shortest_path_to_receptacle_map = self._get_local_distance_map(global_shortest_path_to_receptacle_map)
            channels.append(local_shortest_path_to_receptacle_map)

        # Shortest path distance map
        if self.env.use_shortest_path_map:
            global_shortest_path_map = self._create_global_shortest_path_map()
            local_shortest_path_map = self._get_local_distance_map(global_shortest_path_map)
            channels.append(local_shortest_path_map)

        # History map
        if self.env.use_history_map:
            global_history_map = self._create_global_intention_or_history_map(encoding='history')
            local_history_map = self._get_local_map(global_history_map)
            channels.append(local_history_map)

        # Intention map
        if self.env.use_intention_map:
            global_intention_map = self._create_global_intention_or_history_map(encoding=self.env.intention_map_encoding)
            local_intention_map = self._get_local_map(global_intention_map)
            channels.append(local_intention_map)

        # Baseline intention channels
        if self.env.use_intention_channels:
            intention_channels = self._get_intention_channels()
            channels.extend(intention_channels)

        if save_figures:
            from PIL import Image; import utils  # pylint: disable=import-outside-toplevel
            #print(self.robot.get_position(), self.robot.get_heading())
            output_dir = Path('figures') / 'robot_id_{}'.format(self.robot.id)
            if not output_dir.exists():
                output_dir.mkdir(parents=True)

            def global_map_room_only(global_map):
                crop_width = Mapper.round_up_to_even((self.env.room_length + 2 * Robot.HALF_WIDTH) * Mapper.LOCAL_MAP_PIXELS_PER_METER)
                crop_height = Mapper.round_up_to_even((self.env.room_width + 2 * Robot.HALF_WIDTH) * Mapper.LOCAL_MAP_PIXELS_PER_METER)
                start_i = global_map.shape[0] // 2 - crop_height // 2
                start_j = global_map.shape[1] // 2 - crop_width // 2
                return global_map[start_i:start_i + crop_height, start_j:start_j + crop_width]

            # Environment
            Image.fromarray(self.env.get_camera_image()).save(output_dir / 'env.png')

            def visualize_overhead_map(global_overhead_map, local_overhead_map):
                brightness_scale_factor = 1.33
                global_overhead_map_vis = brightness_scale_factor * global_map_room_only(global_overhead_map)
                local_overhead_map_vis = brightness_scale_factor * local_overhead_map
                return global_overhead_map_vis, local_overhead_map_vis

            # Overhead map
            global_overhead_map_vis, local_overhead_map_vis = visualize_overhead_map(global_overhead_map, local_overhead_map)
            utils.enlarge_image(Image.fromarray(utils.to_uint8_image(global_overhead_map_vis))).save(output_dir / 'global-overhead-map.png')
            utils.enlarge_image(Image.fromarray(utils.to_uint8_image(local_overhead_map_vis))).save(output_dir / 'local-overhead-map.png')

            def visualize_map(overhead_map_vis, distance_map):
                overhead_map_vis = np.stack(3 * [overhead_map_vis], axis=2)
                distance_map_vis = utils.JET[utils.to_uint8_image(distance_map), :]
                return 0.5 * overhead_map_vis + 0.5 * distance_map_vis

            def save_map_visualization(global_map, local_map, suffix, brightness_scale_factor=1):
                global_map_vis = global_map_room_only(global_map)
                global_map_vis = visualize_map(global_overhead_map_vis, brightness_scale_factor * global_map_vis)
                utils.enlarge_image(Image.fromarray(utils.to_uint8_image(global_map_vis))).save(output_dir / 'global-{}.png'.format(suffix))
                local_map = visualize_map(local_overhead_map_vis, brightness_scale_factor * local_map)
                utils.enlarge_image(Image.fromarray(utils.to_uint8_image(local_map))).save(output_dir / 'local-{}.png'.format(suffix))

            # Robot map
            if self.env.use_robot_map:
                save_map_visualization(global_robot_map, local_robot_map, 'robot-map')

            # Shortest path distance to receptacle map
            if self.env.use_shortest_path_to_receptacle_map:
                save_map_visualization(global_shortest_path_to_receptacle_map, local_shortest_path_to_receptacle_map, 'shortest-path-to-receptacle-map', brightness_scale_factor=2)

            # Shortest path distance map
            if self.env.use_shortest_path_map:
                save_map_visualization(global_shortest_path_map, local_shortest_path_map, 'shortest-path-map', brightness_scale_factor=2)

            # History map
            if self.env.use_history_map:
                save_map_visualization(global_history_map, local_history_map, 'history-map')

            # Intention map
            if self.env.use_intention_map:
                save_map_visualization(global_intention_map, local_intention_map, 'intention-map')

            # Baseline intention channels
            if self.env.use_intention_channels:
                for i, channel in enumerate(intention_channels):
                    utils.enlarge_image(Image.fromarray(utils.to_uint8_image(visualize_map(local_overhead_map_vis, np.abs(channel))))).save(output_dir / 'intention-channel{}.png'.format(i))

            # Occupancy map
            if self.env.show_occupancy_maps:
                self.global_occupancy_map.save_figure(output_dir / 'global-occupancy-map.png')

        assert all(channel.dtype == np.float32 for channel in channels)
        return np.stack(channels, axis=2)

    def shortest_path(self, source_position, target_position):
        return self.global_occupancy_map.shortest_path(source_position, target_position)

    def distance_to_receptacle(self, position):
        assert self.env.receptacle_id is not None
        if self.env.use_shortest_path_partial_rewards:
            # Use receptacle as shortest path source for better caching
            return self._shortest_path_distance(self.env.receptacle_position, position)
        return distance(position, self.env.receptacle_position)

    def _shortest_path_distance(self, source_position, target_position):
        return self.global_occupancy_map.shortest_path_distance(source_position, target_position)

    def _get_local_map(self, global_map):
        robot_position, robot_heading = self.robot.get_position(), self.robot.get_heading()
        crop_width = Mapper.round_up_to_even(math.sqrt(2) * Mapper.LOCAL_MAP_PIXEL_WIDTH)
        rotation_angle = 90 - math.degrees(robot_heading)
        pixel_i, pixel_j = Mapper.position_to_pixel_indices(robot_position[0], robot_position[1], global_map.shape)
        crop = global_map[pixel_i - crop_width // 2:pixel_i + crop_width // 2, pixel_j - crop_width // 2:pixel_j + crop_width // 2]
        rotated_crop = rotate_image(crop, rotation_angle, order=0)
        local_map = rotated_crop[
            rotated_crop.shape[0] // 2 - Mapper.LOCAL_MAP_PIXEL_WIDTH // 2:rotated_crop.shape[0] // 2 + Mapper.LOCAL_MAP_PIXEL_WIDTH // 2,
            rotated_crop.shape[1] // 2 - Mapper.LOCAL_MAP_PIXEL_WIDTH // 2:rotated_crop.shape[1] // 2 + Mapper.LOCAL_MAP_PIXEL_WIDTH // 2
        ]
        return local_map

    def _get_local_distance_map(self, global_map):
        local_map = self._get_local_map(global_map)
        local_map -= local_map.min()
        return local_map

    @staticmethod
    def _create_robot_mask(robot_cls, show_lifted_cube=False):
        robot_pixel_width = math.ceil(2 * robot_cls.RADIUS * Mapper.LOCAL_MAP_PIXELS_PER_METER)
        robot_mask = np.zeros((Mapper.LOCAL_MAP_PIXEL_WIDTH, Mapper.LOCAL_MAP_PIXEL_WIDTH), dtype=np.float32)
        start = math.floor(Mapper.LOCAL_MAP_PIXEL_WIDTH / 2 - robot_pixel_width / 2)
        if show_lifted_cube:
            assert robot_cls is LiftingRobot
            cube_pixel_width = math.ceil(VectorEnv.CUBE_WIDTH * Mapper.LOCAL_MAP_PIXELS_PER_METER)

        for i in range(start - cube_pixel_width if show_lifted_cube else start, start + robot_pixel_width):
            for j in range(start, start + robot_pixel_width):
                position_x, position_y = Mapper.pixel_indices_to_position(i, j, robot_mask.shape)
                # Rectangular base
                in_base = abs(position_x) <= Robot.HALF_WIDTH and 0 <= position_y - Robot.BACKPACK_OFFSET <= robot_cls.BASE_LENGTH
                in_backpack = position_x**2 + (position_y - Robot.BACKPACK_OFFSET)**2 <= Robot.HALF_WIDTH ** 2  # Circular backpack
                if in_base or in_backpack:
                    robot_mask[i, j] = 1

                if show_lifted_cube:
                    in_lifted_cube = (abs(position_x) <= VectorEnv.CUBE_WIDTH / 2 and
                        0 <= position_y - (LiftingRobot.END_EFFECTOR_LOCATION + LiftingRobot.LIFTED_CUBE_OFFSET) <= VectorEnv.CUBE_WIDTH)
                    if in_lifted_cube:
                        robot_mask[i, j] = 1

        return robot_mask

    def _create_global_overhead_map(self):
        global_overhead_map = self.global_overhead_map_without_robots.copy()
        global_robot_map_seg = self._create_global_robot_map(seg=True)
        global_overhead_map[global_robot_map_seg > 0] = global_robot_map_seg[global_robot_map_seg > 0]
        assert global_overhead_map.max() <= 1
        return global_overhead_map

    def _create_global_robot_map(self, seg=True):
        global_robot_map = self._create_padded_room_zeros()
        for robot in self.env.robots:
            # Create robot visualization
            robot_vis = self.robot_masks[robot.__class__].copy()
            if seg:
                robot_vis *= self.camera.get_seg_value('robot_group_{}'.format(robot.group_index + 1))
            else:
                if isinstance(robot, LiftingRobot):
                    if robot.lift_state == 'lifting':
                        robot_vis = self.robot_masks['lifting_robot_with_cube'].copy()
                    else:
                        robot_vis *= 0.5

            # Rotate based on robot heading
            rotation_angle = math.degrees(robot.get_heading()) - 90
            rotated = rotate_image(robot_vis, rotation_angle, order=0)

            # Place into global robot map
            robot_position = robot.get_position()
            pixel_i, pixel_j = Mapper.position_to_pixel_indices(robot_position[0], robot_position[1], global_robot_map.shape)
            start_i, start_j = pixel_i - rotated.shape[0] // 2, pixel_j - rotated.shape[1] // 2
            global_robot_map[start_i:start_i + rotated.shape[0], start_j:start_j + rotated.shape[1]] = np.maximum(
                global_robot_map[start_i:start_i + rotated.shape[0], start_j:start_j + rotated.shape[1]], rotated)

        return global_robot_map

    def _create_global_distance_to_receptacle_map(self):
        assert self.env.receptacle_id is not None
        global_map = self._create_padded_room_zeros()
        for i in range(global_map.shape[0]):
            for j in range(global_map.shape[1]):
                pos_x, pos_y = Mapper.pixel_indices_to_position(i, j, global_map.shape)
                global_map[i, j] = distance((pos_x, pos_y), self.env.receptacle_position)
        global_map *= self.env.distance_to_receptacle_map_scale
        return global_map

    def _create_global_shortest_path_to_receptacle_map(self):
        assert self.env.receptacle_id is not None
        global_map = self.global_occupancy_map.shortest_path_image(self.env.receptacle_position)
        global_map[global_map < 0] = global_map.max()
        global_map *= self.env.shortest_path_map_scale
        return global_map

    def _create_global_shortest_path_map(self):
        robot_position = self.robot.get_position()
        global_map = self.global_occupancy_map.shortest_path_image(robot_position)
        global_map[global_map < 0] = global_map.max()
        global_map *= self.env.shortest_path_map_scale
        return global_map

    def _create_global_intention_or_history_map(self, encoding):
        global_intention_map = self._create_padded_room_zeros()
        for robot in self.env.robots:
            if robot is self.robot or robot.is_idle():
                continue

            if encoding == 'circle':
                target_i, target_j = Mapper.position_to_pixel_indices(robot.target_end_effector_position[0], robot.target_end_effector_position[1], global_intention_map.shape)
                global_intention_map[target_i, target_j] = self.env.intention_map_scale
                continue

            if encoding in {'ramp', 'binary', 'line'}:
                waypoint_positions = robot.controller.get_intention_path()
                if encoding == 'line':
                    waypoint_positions = [waypoint_positions[0], waypoint_positions[-1]]
            elif encoding == 'history':
                waypoint_positions = robot.controller.get_history_path()[::-1]

            path_length = 0
            for i in range(1, len(waypoint_positions)):
                source_position = waypoint_positions[i - 1]
                target_position = waypoint_positions[i]
                segment_length = self.env.intention_map_scale * distance(source_position, target_position)

                source_i, source_j = Mapper.position_to_pixel_indices(source_position[0], source_position[1], global_intention_map.shape)
                target_i, target_j = Mapper.position_to_pixel_indices(target_position[0], target_position[1], global_intention_map.shape)
                rr, cc = line(source_i, source_j, target_i, target_j)

                if encoding in {'binary', 'line'}:
                    if i < len(waypoint_positions) - 1:
                        rr, cc = rr[:-1], cc[:-1]
                    global_intention_map[rr, cc] = self.env.intention_map_scale
                elif encoding in {'ramp', 'history'}:
                    line_values = np.clip(np.linspace(1 - path_length, 1 - (path_length + segment_length), len(rr)), 0, 1)
                    if i < len(waypoint_positions) - 1:
                        rr, cc = rr[:-1], cc[:-1]
                        line_values = line_values[:-1]
                    global_intention_map[rr, cc] = np.maximum(global_intention_map[rr, cc], line_values)

                path_length += segment_length

        # Make lines thicker
        if self.env.intention_map_line_thickness > 1:
            global_intention_map = dilation(global_intention_map, self.intention_map_selem)

        return global_intention_map

    def _get_intention_channels(self):
        robot_position, robot_heading = self.robot.get_position(), self.robot.get_heading()
        dists = [distance(robot_position, robot.get_position()) for robot in self.env.robots]

        # Arrange channels in order from closest robot to furthest robot
        channels = []
        for i in np.argsort(dists):
            robot = self.env.robots[i]

            if robot is self.robot:
                continue

            if self.env.intention_channel_encoding == 'spatial':
                global_map = self._create_padded_room_zeros()
                if not robot.is_idle():
                    target_i, target_j = Mapper.position_to_pixel_indices(robot.target_end_effector_position[0], robot.target_end_effector_position[1], global_map.shape)
                    global_map[target_i, target_j] = self.env.intention_map_scale
                    global_map = dilation(global_map, self.intention_map_selem)
                channels.append(self._get_local_map(global_map))

            elif self.env.intention_channel_encoding == 'nonspatial':
                relative_position = (0, 0)
                if not robot.is_idle():
                    dist = distance(robot_position, robot.target_end_effector_position)
                    theta = robot_heading - math.atan2(robot.target_end_effector_position[1] - robot_position[1], robot.target_end_effector_position[0] - robot_position[0])
                    relative_position = (dist * math.sin(theta), dist * math.cos(theta))
                for coord in relative_position:
                    channels.append(self.env.intention_channel_nonspatial_scale * coord * np.ones((Mapper.LOCAL_MAP_PIXEL_WIDTH, Mapper.LOCAL_MAP_PIXEL_WIDTH), dtype=np.float32))

        return channels

    def _create_padded_room_zeros(self):
        return Mapper.create_padded_room_zeros(self.env.room_width, self.env.room_length)

    @staticmethod
    def create_padded_room_zeros(room_width, room_length):
        # Ensure dimensions are even
        return np.zeros((
            Mapper.round_up_to_even(room_width * Mapper.LOCAL_MAP_PIXELS_PER_METER + math.sqrt(2) * Mapper.LOCAL_MAP_PIXEL_WIDTH),
            Mapper.round_up_to_even(room_length * Mapper.LOCAL_MAP_PIXELS_PER_METER + math.sqrt(2) * Mapper.LOCAL_MAP_PIXEL_WIDTH)
        ), dtype=np.float32)

    @staticmethod
    def position_to_pixel_indices(position_x, position_y, image_shape):
        pixel_i = np.floor(image_shape[0] / 2 - position_y * Mapper.LOCAL_MAP_PIXELS_PER_METER).astype(np.int32)
        pixel_j = np.floor(image_shape[1] / 2 + position_x * Mapper.LOCAL_MAP_PIXELS_PER_METER).astype(np.int32)
        pixel_i = np.clip(pixel_i, 0, image_shape[0] - 1)
        pixel_j = np.clip(pixel_j, 0, image_shape[1] - 1)
        return pixel_i, pixel_j

    @staticmethod
    def pixel_indices_to_position(pixel_i, pixel_j, image_shape):
        position_x = ((pixel_j + 0.5) - image_shape[1] / 2) / Mapper.LOCAL_MAP_PIXELS_PER_METER
        position_y = (image_shape[0] / 2 - (pixel_i + 0.5)) / Mapper.LOCAL_MAP_PIXELS_PER_METER
        return position_x, position_y

    @staticmethod
    def round_up_to_even(x):
        return 2 * math.ceil(x / 2)

class OccupancyMap:
    def __init__(self, robot, room_length, room_width, show_map=False):
        self.robot = robot
        self.room_length = room_length
        self.room_width = room_width
        self.show_map = show_map

        # Binary map showing where obstacles are
        self.occupancy_map = self._create_padded_room_zeros().astype(np.uint8)

        # Configuration space for computing shortest paths
        self.configuration_space = None
        self.selem = disk(math.floor(self.robot.RADIUS * Mapper.LOCAL_MAP_PIXELS_PER_METER))
        self.closest_cspace_indices = None

        # Grid graph for computing shortest paths
        self.grid_graph = None

        # Configuration space checking for straight line paths
        self.cspace_thin = None
        self.selem_thin = disk(math.ceil(Robot.HALF_WIDTH * Mapper.LOCAL_MAP_PIXELS_PER_METER))

        # Precompute room mask, which is used to mask out the wall pixels
        self.room_mask = self._create_room_mask()

        if self.show_map:
            import matplotlib.pyplot as plt  # pylint: disable=import-outside-toplevel
            self.plt = plt
            self.plt.ion()
            self.fig_width = self.room_length + 2 * Robot.HALF_WIDTH
            self.fig_height = self.room_width + 2 * Robot.HALF_WIDTH
            figsize = (4 * self.fig_width, 4 * self.fig_height)
            self.fig = self.plt.figure(self.robot.id, figsize=figsize)
            self.free_space_map = self._create_padded_room_zeros().astype(np.uint8)
            self._update_map_visualization()

    def update(self, points, seg, obstacle_seg_value):
        # Incorporate new observation into occupancy map
        augmented_points = np.concatenate([points, np.isclose(seg[:, :, np.newaxis], obstacle_seg_value)], axis=2).reshape(-1, 4)
        obstacle_points = augmented_points[np.isclose(augmented_points[:, 3], 1)]
        pixel_i, pixel_j = Mapper.position_to_pixel_indices(obstacle_points[:, 0], obstacle_points[:, 1], self.occupancy_map.shape)
        self.occupancy_map[pixel_i, pixel_j] = 1
        assert self.occupancy_map.dtype == np.uint8

        # Update configuration space
        self.configuration_space = 1 - np.maximum(1 - self.room_mask, binary_dilation(self.occupancy_map, self.selem).astype(np.uint8))
        self.closest_cspace_indices = distance_transform_edt(1 - self.configuration_space, return_distances=False, return_indices=True)
        self.cspace_thin = 1 - binary_dilation(np.minimum(self.room_mask, self.occupancy_map), self.selem_thin).astype(np.uint8)  # No walls
        assert self.configuration_space.dtype == np.uint8

        # Create a new grid graph with updated configuration space
        self.grid_graph = GridGraph(self.configuration_space)

        if self.show_map:
            free_space_points = augmented_points[np.isclose(augmented_points[:, 3], 0)]
            pixel_i, pixel_j = Mapper.position_to_pixel_indices(free_space_points[:, 0], free_space_points[:, 1], self.free_space_map.shape)
            self.free_space_map[pixel_i, pixel_j] = 1
            self._update_map_visualization()

    def _create_room_mask(self):
        room_mask = self._create_padded_room_zeros().astype(np.uint8)
        room_length_pixels = Mapper.round_up_to_even((self.room_length - 2 * Robot.HALF_WIDTH) * Mapper.LOCAL_MAP_PIXELS_PER_METER)
        room_width_pixels = Mapper.round_up_to_even((self.room_width - 2 * Robot.HALF_WIDTH) * Mapper.LOCAL_MAP_PIXELS_PER_METER)
        start_i = int(room_mask.shape[0] / 2 - room_width_pixels / 2)
        start_j = int(room_mask.shape[1] / 2 - room_length_pixels / 2)
        room_mask[start_i:start_i + room_width_pixels, start_j:start_j + room_length_pixels] = 1
        assert room_mask.dtype == np.uint8
        return room_mask

    def shortest_path(self, source_position, target_position):
        # Convert positions to pixel indices
        source_i, source_j = Mapper.position_to_pixel_indices(source_position[0], source_position[1], self.configuration_space.shape)
        target_i, target_j = Mapper.position_to_pixel_indices(target_position[0], target_position[1], self.configuration_space.shape)

        # Check if there is a straight line path
        rr, cc = line(source_i, source_j, target_i, target_j)
        if (1 - self.cspace_thin[rr, cc]).sum() == 0:
            return [source_position, target_position]

        # Run SPFA
        source_i, source_j = self._closest_valid_cspace_indices(source_i, source_j)
        target_i, target_j = self._closest_valid_cspace_indices(target_i, target_j)
        path_pixel_indices = self.grid_graph.shortest_path((source_i, source_j), (target_i, target_j))

        # Convert pixel indices back to positions
        path = []
        for i, j in path_pixel_indices:
            position_x, position_y = Mapper.pixel_indices_to_position(i, j, self.configuration_space.shape)
            path.append((position_x, position_y, 0))

        if len(path) < 2:
            path = [source_position, target_position]
        else:
            path[0] = source_position
            path[-1] = target_position

        return path

    def shortest_path_distance(self, source_position, target_position):
        source_i, source_j = Mapper.position_to_pixel_indices(source_position[0], source_position[1], self.configuration_space.shape)
        target_i, target_j = Mapper.position_to_pixel_indices(target_position[0], target_position[1], self.configuration_space.shape)
        source_i, source_j = self._closest_valid_cspace_indices(source_i, source_j)
        target_i, target_j = self._closest_valid_cspace_indices(target_i, target_j)
        return self.grid_graph.shortest_path_distance((source_i, source_j), (target_i, target_j)) / Mapper.LOCAL_MAP_PIXELS_PER_METER

    def shortest_path_image(self, position):
        target_i, target_j = Mapper.position_to_pixel_indices(position[0], position[1], self.configuration_space.shape)
        target_i, target_j = self._closest_valid_cspace_indices(target_i, target_j)
        return self.grid_graph.shortest_path_image((target_i, target_j)) / Mapper.LOCAL_MAP_PIXELS_PER_METER

    def save_figure(self, output_path):
        assert self.show_map
        self.fig.savefig(output_path, bbox_inches='tight', pad_inches=0)

    def _closest_valid_cspace_indices(self, i, j):
        return self.closest_cspace_indices[:, i, j]

    def _create_padded_room_zeros(self):
        return Mapper.create_padded_room_zeros(self.room_width, self.room_length)

    def _update_map_visualization(self):
        # Create map visualization
        occupancy_map_vis = self._create_padded_room_zeros() + 0.5
        occupancy_map_vis[self.free_space_map == 1] = 1
        occupancy_map_vis[self.occupancy_map == 1] = 0

        # Show map visualization
        self.fig.clf()
        self.fig.add_axes((0, 0, 1, 1))
        ax = self.fig.gca()
        ax.axis('off')
        ax.axis([-self.fig_width / 2, self.fig_width / 2, -self.fig_height / 2, self.fig_height / 2])
        height, width = occupancy_map_vis.shape
        height, width = height / Mapper.LOCAL_MAP_PIXELS_PER_METER, width / Mapper.LOCAL_MAP_PIXELS_PER_METER
        ax.imshow(255.0 * occupancy_map_vis, extent=(-width / 2, width / 2, -height / 2, height / 2), cmap='gray', vmin=0, vmax=255.0)

        # Show waypoint positions
        if self.robot.waypoint_positions is not None:
            waypoint_positions = np.array(self.robot.waypoint_positions)
            ax.plot(waypoint_positions[:, 0], waypoint_positions[:, 1], color='r', marker='.')

        # Show target end effector position
        if self.robot.target_end_effector_position is not None:
            ax.plot(self.robot.target_end_effector_position[0], self.robot.target_end_effector_position[1], color='r', marker='x')

        # Update display
        self.plt.pause(0.001)

def distance(p1, p2):
    return math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

def orientation_to_heading(o):
    # Note: Only works for z-axis rotations
    return 2 * math.acos(math.copysign(1, o[2]) * o[3])

def heading_to_orientation(h):
    return pybullet.getQuaternionFromEuler((0, 0, h))

def restrict_heading_range(h):
    return (h + math.pi) % (2 * math.pi) - math.pi

def heading_difference(h1, h2):
    return restrict_heading_range(h2 - h1)

def dot(a, b):
    return a[0] * b[0] + a[1] * b[1]
