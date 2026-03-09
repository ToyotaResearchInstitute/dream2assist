# Shared control racing environment using epic's car dynamics and physics engine.
# Based on the Box2D Car class.

from copy import deepcopy
import math
import pandas as pd
import random
import torch
from typing import Optional, Union, Mapping

from scipy.spatial.transform import Rotation
import numpy as np

import gymnasium as gym
from gymnasium import spaces
import gymnasium.envs.box2d.car_dynamics as car_dynamics
from gymnasium.envs.box2d.car_dynamics import Car
from gymnasium.error import DependencyNotInstalled
from gymnasium.utils import EzPickle

try:
    import Box2D
    from Box2D.b2 import contactListener, fixtureDef, polygonShape
except ImportError as e:
    raise DependencyNotInstalled(
        "Box2D is not installed, run `pip install gymnasium[box2d]`"
    ) from e

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled(
        "pygame is not installed, run `pip install gymnasium[box2d]`"
    ) from e

from epic_car import EpicCar
from util import wrap_angle, normalize_track_columns


DEVICE = "cuda:0"

# Set to tiny scale, to effectively skip videos when using proprioceptive observations.
VID_SCALE = 100
STATE_W = 64  # less than Atari 160x192
STATE_H = 64
VIDEO_W = 600 // VID_SCALE
VIDEO_H = 400 // VID_SCALE
WINDOW_W = 1000
WINDOW_H = 800

SCALE = 6.0  # Track scale
TRACK_RAD = 900 / SCALE  # Track is heavily morphed circle with this radius
PLAYFIELD = 6000 / SCALE  # Game over boundary
FPS = 50  # Frames per second
ZOOM = 0.02  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE

ADO_STATE_OBS_LOOKAHEAD = 20
# Offset of tiles in state observation. 0 = starting with the tile closest to the car
ADO_STATE_OBS_SHIFT = 2
STATE_OBS_LOOKAHEAD = 400  # Number of tiles to look ahead in the state observation
# Offset of tiles in state observation. 0 = starting with the tile closest to the car
STATE_OBS_SHIFT = 5
STATE_OBS_STEP_SIZE = 10  # Stride when selecting tiles for state observation

# Specify different car colors
CAR_COLORS = [
    (0.8, 0.0, 0.0),
    (0.0, 0.0, 0.8),
    (0.0, 0.8, 0.0),
    (0.0, 0.8, 0.8),
    (0.8, 0.8, 0.8),
    (0.0, 0.0, 0.0),
    (0.8, 0.0, 0.8),
    (0.8, 0.8, 0.0),
]

# Distance between cars
LINE_SPACING = 50  # Starting distance between each pair of cars
LATERAL_SPACING = 1  # Starting side distance between pairs of cars

# Pure-pursuit params.
K_LOOKAHEAD = 20  # segments ahead
K_SPEED = 20
K_PROP = 1e-1
K_WB = 2.9  # wheel base of vehicle

# Unallocated action space indices
STEER_INDEX = 0
THROTTLE_INDEX = 1
BRAKE_INDEX = 2

# Default values for starting position at a particular track location, with some randomization
DEFAULT_START_TRACK_FRAC = 0.8
DEFAULT_START_RANGE = 100.0


def proportional_control(current_speed, target_speed):
    a = K_PROP * (target_speed - current_speed)
    return a


def pure_pursuit(targets, yaw, heading_gain=0.4):
    index = min(K_LOOKAHEAD, len(targets) - 1)
    target = targets[index]
    tx = target[0]
    ty = target[1]
    alpha = heading_gain * (math.atan2(tx, -ty) - yaw)
    # print(f"tx: {tx}, ty: {ty}, yaw: {yaw}, alpha: {alpha}")
    delta = math.atan2(2.0 * K_WB * math.sin(alpha) / (np.linalg.norm(target) + 1e-6), 1.0)
    return delta


class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent, pass_bonus=None, verbose=False):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent
        self.pass_bonus = pass_bonus
        self.verbose = verbose

    def BeginContact(self, contact):
        self._contact(contact, True)

    def EndContact(self, contact):
        self._contact(contact, False)

    def _contact(self, contact, begin):
        tile = None
        obj = None
        u1 = contact.fixtureA.body.userData
        u2 = contact.fixtureB.body.userData
        if u1 and "road_friction" in u1.__dict__:
            tile = u1
            obj = u2
        if u2 and "road_friction" in u2.__dict__:
            tile = u2
            obj = u1
        if not tile:
            return

        # inherit tile color from env
        tile.color[:] = self.env.road_color
        if not obj or "tiles" not in obj.__dict__:
            return
        if begin:
            obj.tiles.add(tile)
            if not tile.road_visited[obj.car_id]:
                tile.road_visited[obj.car_id] = True
                self.env.tile_visited_count[obj.car_id] += 1
                self.env.on_same_tile_count[obj.car_id] = 0

                # The reward is dampened on tiles that have been visited already.
                past_visitors = sum(tile.road_visited) - 1
                reward_factor = 1
                if self.pass_bonus is not None:
                    if self.pass_bonus:
                        reward_factor = 1 - (past_visitors / self.env.num_agents)
                self.env.reward[obj.car_id] += reward_factor * 1000.0 / len(self.env.track)
                if self.verbose:
                    print(
                        f"  car {obj.car_id} incrementing reward by {reward_factor * 1000.0 / len(self.env.track)}"
                    )

                # Lap is considered completed if enough % of the track was covered
                if (
                    tile.idx == 0
                    and self.env.tile_visited_count[obj.car_id] / len(self.env.track)
                    > self.lap_complete_percent
                ):
                    self.env.new_lap = True
            else:
                self.env.on_same_tile_count[obj.car_id] += 1
        else:
            obj.tiles.remove(tile)


class EpicDynamicsSharedRacing(gym.Env, EzPickle):
    """
    Shared control racing environment using EPIC's car dynamics and physics engine.
    TODO(jon) Reduce overlap with the Box2D environment.

    Args:
        config (object): The configuration object for the environment.
        render_mode (str): The render mode for the environment.
        verbose (bool): Whether to print debug information.
        lap_complete_percent (float): The percentage of the track that must be covered to complete a lap.
        domain_randomize (bool): Whether to randomize the domain.
        continuous (bool): Whether the environment is continuous.
        pixel_obs (bool): Whether to use pixel observations.
        num_agents (int): The number of agents in the environment.
        track_csv (str): The path to the track CSV file.
        dreamer_policy (object): The Dreamer policy to use.
        start_position (Mapping): The starting position for the environment.
        ado_rails (object): The ado rails for the environment.
        ado_names (list): The names of the ados in the environment.

    Attributes:
        metadata (dict): The metadata for the environment.
    """

    metadata = {
        "render_modes": ["rgb_array", "state_pixels", "no_render", "human", "bev"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        config,
        render_mode: Optional[str] = None,
        verbose: bool = False,
        lap_complete_percent: float = 0.95,
        domain_randomize: bool = False,
        continuous: bool = False,
        pixel_obs: bool = False,
        num_agents: int = 2,
        track_csv: Optional[str] = None,
        dreamer_policy: Optional[object] = None,
        start_position: Optional[Mapping] = None,
        ado_rails: Optional[object] = None,
        ado_names: Optional[list] = None,
        ado_trigger_groups: Optional[list] = None,
        ado_speed_factors: Optional[list] = None,
        pct_track_to_visit: Optional[int] = None,
    ):
        EzPickle.__init__(
            self,
            render_mode,
            verbose,
            lap_complete_percent,
            domain_randomize,
            continuous,
            pixel_obs,
            num_agents,
        )
        self.num_agents = num_agents
        self.pixel_obs = pixel_obs
        self.continuous = continuous
        self.domain_randomize = domain_randomize
        # TODO: Vectorize this per the number of agents?
        self.lap_complete_percent = lap_complete_percent
        self.randomize_ego_start = config.randomize_ego_start if config is not None else False
        self.pct_track_to_visit = pct_track_to_visit
        self.episode_tiles_visited = None
        self._init_colors()

        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent, verbose)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: Optional[pygame.Surface] = None

        # These need to get set externally upon reset()
        self.ado_rails = ado_rails
        self.ado_names = ado_names
        self.ado_trigger_groups = ado_trigger_groups
        self.ado_speed_factors = [1.0] * (num_agents - 1)
        if ado_speed_factors is not None:
            indices = [
                [i for i, t in enumerate(ado_trigger_groups) if n in t][0] for n in ado_names
            ]
            self.ado_speed_factors = [ado_speed_factors[i] for i in indices]

        self.surf = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        self.cars = [None] * num_agents
        self.car_order = None  # Determines starting positions of cars
        self.reward = np.zeros(num_agents)
        self.prev_reward = np.zeros(num_agents)
        self.tile_visited_count = [0] * num_agents
        self.on_same_tile_count = [0] * num_agents
        self.verbose = verbose
        self.new_lap = False
        self.fd_tile = fixtureDef(shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]))
        # Storage for road features that are part of the state.
        self.state_obs_features = [None] * num_agents
        self.track = None
        self.track_left = None
        self.track_right = None
        self.ado_state = None
        self.ego_decision = None
        self.ai_decision = None
        self.x_ddm = None
        self.ai_acceptance = None
        self.is_within_range = False
        self.track_csv = track_csv
        self.track_df = pd.read_csv(track_csv)
        self.road_poly = []
        self.ego_has_passed_ado_count = 0
        self.ego_has_already_passed_ado = False
        self.verbose = verbose
        self.step_time = 0.1
        self.start_dict = {}
        self.step_time = 0.1  # Simulation discretization time step for EpicCar.

        self.ego_index = 0
        self.ado_indices = [i for i in range(1, num_agents)]
        self.num_ados = len(self.ado_indices)
        self.max_num_ados = config.max_num_ados if config is not None else self.num_ados
        self.max_num_ados = min(self.max_num_ados, self.num_ados)
        self.step_size = None  # Note: will be set when the ego car is created.

        self.use_pure_pursuit = config.use_pure_pursuit if config is not None else False
        self.use_race_line = config.use_race_line if config is not None else False
        self.rotate_ado_90degrees = config.rotate_ado_90degrees if config is not None else False
        self.ado_trigger_radius_lb = config.ado_trigger_radius_lb if config is not None else 15.0
        self.ado_trigger_radius_ub = config.ado_trigger_radius_ub if config is not None else 40.0
        self.ado_trigger_radius = random.uniform(
            self.ado_trigger_radius_lb, self.ado_trigger_radius_ub
        )
        self.include_ego_position_in_state = (
            config.include_ego_position_in_state if config is not None else False
        )
        self.use_relative_heading_in_state = (
            config.use_relative_heading_in_state if config is not None else False
        )
        self.randomize_road_position = (
            config.randomize_road_position if config is not None and hasattr(config, 'randomize_road_position') else False
        )
        self.road_offset_lateral = (
            config.road_offset_lateral if config is not None and hasattr(config, 'road_offset_lateral') else 0.0
        )
        self.road_offset_angular = (
            config.road_offset_angular if config is not None and hasattr(config, 'road_offset_angular') else 0.0
        )

        self.prev_ego_position = None
        self.ego_velocity = np.array([0.0, 0.0])

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised however this is not possible here so ignore

        # We're training only for the ego vehicle here; its observation space should include the ado's state.
        if self.continuous:
            # Define upper and lower bounds on the action space.
            # [steer, gas, brake]
            action_lb = np.array([-1, 0, 0]).astype(np.float64)
            action_ub = np.array([+1, +1, +1]).astype(np.float64)
            self.action_space = spaces.Box(action_lb, action_ub)
            self.actions = ["STEER_A", "GAS_A", "BRAKE_A"]
        else:
            raise TypeError("Must be a continuous action space!")

        self.render_mode = "state_pixels" if render_mode is not "no_render" else render_mode
        self.state_length = 0
        if self.pixel_obs:
            self.observation_space = spaces.Dict(
                image=spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8),
            )
        else:
            # Makes train mode more efficient.
            self.render_mode = ("rgb_array" if render_mode is None else render_mode)
            # Define upper and lower bounds on the observation space.
            self.state_length = 5  # (x, y, angle, v_x, v_y)
            if self.include_ego_position_in_state:
                # For each car, we have: x, y, angle, v_x, v_y, tile_x1, tile_y1, tile_x2, tile_y2, ..., tile_xN, tile_yN, ado1_angle, ado1_x, ado1_y, ado1_rel_v_x, ado1_rel_v_y, ..., adoM_angle, adoM_x, adoM_y, adoM_rel_v_x, adoM_rel_v_y
                obs_lb = np.array(
                    (
                        *[-PLAYFIELD] * 2,
                        -2.0 * math.pi,
                        0.0,
                        0.0,
                        *[-PLAYFIELD]
                        * (2 * int(np.ceil(STATE_OBS_LOOKAHEAD / STATE_OBS_STEP_SIZE)))
                        * 3,
                    )
                )
                for _ in range(self.max_num_ados):
                    obs_lb = np.array((*obs_lb, *[-PLAYFIELD] * 2, -2.0 * math.pi, 0.0, 0.0))

                obs_ub = np.array(
                    (
                        *[PLAYFIELD] * 2,
                        2.0 * math.pi,
                        100.0,
                        100.0,
                        *[PLAYFIELD]
                        * (2 * int(np.ceil(STATE_OBS_LOOKAHEAD / STATE_OBS_STEP_SIZE)))
                        * 3,
                    )
                )
                for _ in range(self.max_num_ados):
                    obs_ub = np.array((*obs_ub, *[PLAYFIELD] * 2, 2.0 * math.pi, 100.0, 100.0))
            else:
                # For each car, we have: angle, v_x, v_y, tile_x1, tile_y1, tile_x2, tile_y2, ..., tile_xN, tile_yN, ado1_angle, ado1_x, ado1_y, ado1_rel_v_x, ado1_rel_v_y, ..., adoM_angle, adoM_x, adoM_y, adoM_rel_v_x, adoM_rel_v_y
                obs_lb = np.array(
                    (
                        -2.0 * math.pi,
                        0.0,
                        0.0,
                        *[-PLAYFIELD]
                        * (2 * int(np.ceil(STATE_OBS_LOOKAHEAD / STATE_OBS_STEP_SIZE)))
                        * 3,
                    )
                )
                for _ in range(self.max_num_ados):
                    obs_lb = np.array((*obs_lb, *[-PLAYFIELD] * 2, -2.0 * math.pi, 0.0, 0.0))

                obs_ub = np.array(
                    (
                        2.0 * math.pi,
                        100.0,
                        100.0,
                        *[PLAYFIELD]
                        * (2 * int(np.ceil(STATE_OBS_LOOKAHEAD / STATE_OBS_STEP_SIZE)))
                        * 3,
                    )
                )
                for _ in range(self.max_num_ados):
                    obs_ub = np.array((*obs_ub, *[PLAYFIELD] * 2, 2.0 * math.pi, 100.0, 100.0))
            self.observation_space = spaces.Dict(
                image=spaces.Box(low=0, high=255, shape=(1, 1, 3), dtype=np.uint8),
                state=spaces.Box(low=obs_lb, high=obs_ub),
            )

        # Note: The above observation spaces should not account for the actions of other agents. These will be
        # automatically appended to the observation space in the wrapper.

    @property
    def ego_car(self):
        assert self.cars[self.ego_index] is not None
        return self.cars[self.ego_index]

    def ado_car(self, i):
        assert self.cars[self.ado_indices[i]] is not None
        return self.cars[self.ado_indices[i]]

    def init_actions(self):
        return np.zeros(len(self.action_space.low)), np.zeros(len(self.action_space.low))

    def reset_ado_rails(self, ado_rails, ado_trigger_groups=None, ado_speed_factors=None):
        self.ado_rails = ado_rails
        self.ado_trigger_groups = ado_trigger_groups
        if ado_speed_factors is not None:
            indices = [
                [i for i, t in enumerate(ado_trigger_groups) if n in t][0] for n in self.ado_names
            ]
            self.ado_speed_factors = [ado_speed_factors[i] for i in indices]
        self.reset_ado_positions()

    def reset_ado_positions(self):
        # Always call this after setting self.ado_rails and self.ado_names
        ado_start_dict = dict.fromkeys(self.ado_names, {})
        for name in self.ado_names:
            quat = Rotation.from_quat(
                [
                    self.ado_rails[name + "_uvs_orientation_x"][0],
                    self.ado_rails[name + "_uvs_orientation_y"][0],
                    self.ado_rails[name + "_uvs_orientation_z"][0],
                    self.ado_rails[name + "_uvs_orientation_w"][0],
                ],
                scalar_first=False,
            )
            if self.rotate_ado_90degrees:
                rot = Rotation.from_euler("z", -90, degrees=True)
                quat = rot * quat  # Rotate the quaternion to match the Box2D coordinate system.
            q = quat.as_quat(scalar_first=False)  # Convert to quaternion format (x, y, z, w)
            ado_start_dict[name] = {
                "x": self.ado_rails[name + "_uvs_x"][0],
                "y": self.ado_rails[name + "_uvs_y"][0],
                "z": self.ado_rails[name + "_uvs_z"][0],
                "quaternion": q,
            }
            print(f"Setting ado {name} start position to {ado_start_dict[name]}")
        self.start_dict.update(ado_start_dict)

    def _destroy(self):
        if not self.road:
            return
        for t in self.road:
            self.world.DestroyBody(t)
        self.road = []
        for car in self.cars:
            assert car is not None
            car.destroy()

    def _init_colors(self):
        if self.domain_randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20
        else:
            # default colours
            self.road_color = np.array([102, 102, 102])
            self.bg_color = np.array([102, 204, 102])
            self.grass_color = np.array([102, 230, 102])

    def _reinit_colors(self, randomize):
        assert self.domain_randomize, "domain_randomize must be True to use this function."

        if randomize:
            # domain randomize the bg and grass colour
            self.road_color = self.np_random.uniform(0, 210, size=3)

            self.bg_color = self.np_random.uniform(0, 210, size=3)

            self.grass_color = np.copy(self.bg_color)
            idx = self.np_random.integers(3)
            self.grass_color[idx] += 20

    def _create_track_from_csv(self):
        df = self.track_df

        # Normalize column names to handle different CSV formats using shared utility
        df, _ = normalize_track_columns(df, target_format='lowercase')

        # Add path_z if not present (default to 0)
        if 'path_z' not in df.columns:
            df['path_z'] = 0.0

        track = np.array(
            [
                df.loc[:, "left_edge_x"],
                df.loc[:, "left_edge_y"],
                df.loc[:, "right_edge_x"],
                df.loc[:, "right_edge_y"],
                df.loc[:, "path_x"],
                df.loc[:, "path_y"],
                df.loc[:, "path_z"],
            ]
        )
        track = track[:, :-1].T  # The final entry is just a repeat of the first one.
        # Create tiles
        self.road = []
        self.track_left = []
        self.track_right = []
        self.track = []
        for i in range(len(track)):
            road1_l = track[i, :2]
            road1_r = track[i, 2:4]
            road2_l = track[i - 1, :2]
            road2_r = track[i - 1, 2:4]
            alpha = 0.0
            if self.use_race_line:
                ref_line1 = track[i, 4:5]
                ref_line2 = track[i - 1, 4:5]
            else:
                ref_line1 = (road1_l + road1_r) / 2
                ref_line2 = (road2_l + road2_r) / 2
            beta = (
                math.atan((ref_line1[1] - ref_line2[1]) / (ref_line1[0] - ref_line2[0]))
                + math.pi / 2
            )
            self.track.append((alpha, beta, *ref_line1))
            self.track_left.append(road1_l)
            self.track_right.append(road1_r)

            vertices = [tuple(road1_l), tuple(road1_r), tuple(road2_r), tuple(road2_l)]
            self.fd_tile.shape.vertices = vertices

            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = [False] * self.num_agents
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
        return True

    def _create_random_track(self):
        CHECKPOINTS = 12

        # Create checkpoints
        checkpoints = []
        for c in range(CHECKPOINTS):
            noise = self.np_random.uniform(0, 2 * math.pi * 1 / CHECKPOINTS)
            alpha = 2 * math.pi * c / CHECKPOINTS + noise
            rad = self.np_random.uniform(TRACK_RAD / 3, TRACK_RAD)

            if c == 0:
                alpha = 0
                rad = 1.5 * TRACK_RAD
            if c == CHECKPOINTS - 1:
                alpha = 2 * math.pi * c / CHECKPOINTS
                self.start_alpha = 2 * math.pi * (-0.5) / CHECKPOINTS
                rad = 1.5 * TRACK_RAD

            checkpoints.append((alpha, rad * math.cos(alpha), rad * math.sin(alpha)))
        self.road = []

        # Go from one checkpoint to another to create track
        x, y, beta = 1.5 * TRACK_RAD, 0, 0
        dest_i = 0
        laps = 0
        track = []
        no_freeze = 2500
        visited_other_side = False
        while True:
            alpha = math.atan2(y, x)
            if visited_other_side and alpha > 0:
                laps += 1
                visited_other_side = False
            if alpha < 0:
                visited_other_side = True
                alpha += 2 * math.pi

            while True:  # Find destination from checkpoints
                failed = True

                while True:
                    dest_alpha, dest_x, dest_y = checkpoints[dest_i % len(checkpoints)]
                    if alpha <= dest_alpha:
                        failed = False
                        break
                    dest_i += 1
                    if dest_i % len(checkpoints) == 0:
                        break

                if not failed:
                    break

                alpha -= 2 * math.pi
                continue

            r1x = math.cos(beta)
            r1y = math.sin(beta)
            p1x = -r1y
            p1y = r1x
            dest_dx = dest_x - x  # vector towards destination
            dest_dy = dest_y - y
            # destination vector projected on rad:
            proj = r1x * dest_dx + r1y * dest_dy
            while beta - alpha > 1.5 * math.pi:
                beta -= 2 * math.pi
            while beta - alpha < -1.5 * math.pi:
                beta += 2 * math.pi
            prev_beta = beta
            proj *= SCALE
            if proj > 0.3:
                beta -= min(TRACK_TURN_RATE, abs(0.001 * proj))
            if proj < -0.3:
                beta += min(TRACK_TURN_RATE, abs(0.001 * proj))
            x += p1x * TRACK_DETAIL_STEP
            y += p1y * TRACK_DETAIL_STEP
            track.append((alpha, prev_beta * 0.5 + beta * 0.5, x, y))
            if laps > 4:
                break
            no_freeze -= 1
            if no_freeze == 0:
                break

        # Find closed loop range i1..i2, first loop should be ignored, second is OK
        i1, i2 = -1, -1
        i = len(track)
        while True:
            i -= 1
            if i == 0:
                return False  # Failed
            pass_through_start = (
                track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
            )
            if pass_through_start and i2 == -1:
                i2 = i
            elif pass_through_start and i1 == -1:
                i1 = i
                break
        if self.verbose:
            print("Track generation: %i..%i -> %i-tiles track" % (i1, i2, i2 - i1))
        assert i1 != -1
        assert i2 != -1

        track = track[i1 : i2 - 1]

        first_beta = track[0][1]
        first_perp_x = math.cos(first_beta)
        first_perp_y = math.sin(first_beta)
        # Length of perpendicular jump to put together head and tail
        well_glued_together = np.sqrt(
            np.square(first_perp_x * (track[0][2] - track[-1][2]))
            + np.square(first_perp_y * (track[0][3] - track[-1][3]))
        )
        if well_glued_together > TRACK_DETAIL_STEP:
            return False

        # Red-white border on hard turns
        border = [False] * len(track)
        for i in range(len(track)):
            good = True
            oneside = 0
            for neg in range(BORDER_MIN_COUNT):
                beta1 = track[i - neg - 0][1]
                beta2 = track[i - neg - 1][1]
                good &= abs(beta1 - beta2) > TRACK_TURN_RATE * 0.2
                oneside += np.sign(beta1 - beta2)
            good &= abs(oneside) == BORDER_MIN_COUNT
            border[i] = good
        for i in range(len(track)):
            for neg in range(BORDER_MIN_COUNT):
                border[i - neg] |= border[i]

        # Create tiles
        self.track_left = []
        self.track_right = []
        for i in range(len(track)):
            alpha1, beta1, x1, y1 = track[i]
            alpha2, beta2, x2, y2 = track[i - 1]
            road1_l = (
                x1 - TRACK_WIDTH * math.cos(beta1),
                y1 - TRACK_WIDTH * math.sin(beta1),
            )
            road1_r = (
                x1 + TRACK_WIDTH * math.cos(beta1),
                y1 + TRACK_WIDTH * math.sin(beta1),
            )
            road2_l = (
                x2 - TRACK_WIDTH * math.cos(beta2),
                y2 - TRACK_WIDTH * math.sin(beta2),
            )
            road2_r = (
                x2 + TRACK_WIDTH * math.cos(beta2),
                y2 + TRACK_WIDTH * math.sin(beta2),
            )
            self.track_left.append(road1_l)
            self.track_right.append(road1_r)

            vertices = [road1_l, road1_r, road2_r, road2_l]
            self.fd_tile.shape.vertices = vertices
            t = self.world.CreateStaticBody(fixtures=self.fd_tile)
            t.userData = t
            c = 0.01 * (i % 3) * 255
            t.color = self.road_color + c
            t.road_visited = [False] * self.num_agents
            t.road_friction = 1.0
            t.idx = i
            t.fixtures[0].sensor = True
            self.road_poly.append(([road1_l, road1_r, road2_r, road2_l], t.color))
            self.road.append(t)
            if border[i]:
                side = np.sign(beta2 - beta1)
                b1_l = (
                    x1 + side * TRACK_WIDTH * math.cos(beta1),
                    y1 + side * TRACK_WIDTH * math.sin(beta1),
                )
                b1_r = (
                    x1 + side * (TRACK_WIDTH + BORDER) * math.cos(beta1),
                    y1 + side * (TRACK_WIDTH + BORDER) * math.sin(beta1),
                )
                b2_l = (
                    x2 + side * TRACK_WIDTH * math.cos(beta2),
                    y2 + side * TRACK_WIDTH * math.sin(beta2),
                )
                b2_r = (
                    x2 + side * (TRACK_WIDTH + BORDER) * math.cos(beta2),
                    y2 + side * (TRACK_WIDTH + BORDER) * math.sin(beta2),
                )
                self.road_poly.append(
                    (
                        [b1_l, b1_r, b2_r, b2_l],
                        (255, 255, 255) if i % 2 == 0 else (255, 0, 0),
                    )
                )
        self.track = track
        return True

    def get_ado_obs(self, i):
        # TODO (jon): There appears to no longer be a need to construct an observation at all.  We only need the "state" vector from this function.
        track_xy = np.array(self.track)[:, 2:4]
        left_xy = np.array(self.track_left)
        right_xy = np.array(self.track_right)

        distances = np.linalg.norm(track_xy - self.ado_car(i).hull.position, axis=1)
        next_tile_idx = np.argmin(distances) + ADO_STATE_OBS_SHIFT
        obs_tile_center_ids = (np.array(range(ADO_STATE_OBS_LOOKAHEAD)) + next_tile_idx) % len(
            track_xy
        )

        ado_states = [
            [self.ado_car(i).hull.angle, *self.ado_car(i).hull.position]
            for i in range(self.num_ados)
        ]
        for i in range(self.num_ados):
            rel_track_xy = self._transform_points(ado_states[i], track_xy[obs_tile_center_ids, :])
            rel_left_xy = self._transform_points(ado_states[i], left_xy[obs_tile_center_ids, :])
            rel_right_xy = self._transform_points(ado_states[i], right_xy[obs_tile_center_ids, :])
            rel_obs_features = np.stack([rel_track_xy, rel_left_xy, rel_right_xy])
            ado_states[i].extend(rel_obs_features.flatten())

        pixels_state = {
            "image": np.zeros((self.num_agents, 1, 1, 3)),
            "state": np.array((ado_states)),
            "reward": 0,
            "is_terminal": False,
            "is_first": False,
        }
        return [pixels_state]

    def pure_pursuit(self, i, obs, targets=None, target_speed=K_SPEED):
        # Assumes shared_control_racing obs is [angle, position, flattened_features]
        # [position_x, position_y, angle]
        state = [*obs["state"][0][i][1:3], obs["state"][0][i][0]]
        if self.verbose:
            print(f"ado state {state}")
        # Reshape flattened_features to a 3 x 2 x N array
        rel_features = obs["state"][0][i][3:].reshape(3, -1, 2)
        if targets is None:
            targets = rel_features[0]
        yaw = wrap_angle(state[2])  # Wrap yaw to [-pi, pi]
        delta = pure_pursuit(targets, yaw)
        speed = np.sqrt(
            np.square(self.ado_car(i).hull.linearVelocity[0])
            + np.square(self.ado_car(i).hull.linearVelocity[1])
        )
        if self.verbose:
            print(f"ado speed: {speed}, target speed: {target_speed}, delta: {delta}")
        a = (
            proportional_control(speed, 1.0 * target_speed) if target_speed > 0.1 else -1.0
        )  # Full brake if speed is small.
        action = {
            "action": torch.tensor([[np.clip(delta, -1, 1), np.clip(a, 0, 1), np.clip(-a, 0, 1)]])
        }
        if self.verbose:
            print(
                f" action {action['action']}, speed {speed}, dist to c-line {min(np.linalg.norm(targets - self.ado_car(i).hull.position, axis=1))}"
            )
        return action

    def step_ado(self, i):
        # Initialize or unpack simulation state.
        # Step agent.
        obs = self.get_ado_obs(i)
        done = np.array([False])
        obs = {k: np.stack([obs[0][k]]) for k in obs[0]}
        if self.use_pure_pursuit:
            if self.ado_rails is not None:
                name = self.ado_names[i]
                ado_xy = np.stack(
                    [self.ado_rails[name + "_uvs_x"], self.ado_rails[name + "_uvs_y"]]
                ).T
                ado_speeds = self.ado_rails[name + "_uvs_vx"]
                ado_state = [self.ado_car(i).hull.angle, *self.ado_car(i).hull.position]
                rel_ado_tracks = self._transform_points(ado_state, ado_xy)
                nearest_index = np.argmin(
                    np.linalg.norm(ado_xy - self.ado_car(i).hull.position, axis=1)
                )
                rel_ado_tracks_ahead = rel_ado_tracks[nearest_index:]
                ado_speed = ado_speeds[nearest_index] * self.ado_speed_factors[i]
                if self.verbose:
                    print(
                        f"target ado_speed: {ado_speed}, nearest ado_xy: {ado_xy[nearest_index]}"
                    )  # , ado_rel_tracks: {rel_ado_tracks_ahead}")
                if len(rel_ado_tracks_ahead) == 0:
                    # If all ados are behind us, set the relative tracks ahead
                    # to the last of the relative tracks
                    rel_ado_tracks_ahead = rel_ado_tracks[-1]
                action = self.pure_pursuit(i, obs, rel_ado_tracks_ahead, ado_speed)
                if self.ado_trigger_groups is not None:
                    group = [g for g in self.ado_trigger_groups if name in g][
                        0
                    ]  # We assume ados in the trigger groups are mutually exclusive, hence assume the list has one element.
                    group_indices = [i for i in range(self.num_ados) if self.ado_names[i] in group]
                    if self.verbose:
                        print(
                            f"ado {name} checking trigger group {group}: {[np.linalg.norm(np.array(self.ego_car.hull.position) - self.ado_car(index).hull.position) > self.ado_trigger_radius for index in group_indices]}"
                        )
                    if all(
                        [
                            np.linalg.norm(
                                np.array(self.ego_car.hull.position)
                                - self.ado_car(index).hull.position
                            )
                            > self.ado_trigger_radius
                            for index in group_indices
                        ]
                    ):
                        action = {"action": torch.tensor([[0.0, 0.0, 1.0]])}  # Full brake
                        if self.verbose:
                            print(f"ado {name} braking for trigger group {group}")
                if self.verbose:
                    print(f"ado action: {action['action']}")
            else:
                action = self.pure_pursuit(obs)
        else:
            action, self.ado_state = self.ado_agent(obs, done, self.ado_state)
        if isinstance(action, dict):
            action = {k: np.array(action[k][0].detach().cpu()) for k in action}
        else:
            action = np.array(action)
        return action

    def do_randomize_road_position(self, x, y, yaw):
        """
        Randomize the vehicle's position along the road normal.
        This is useful for domain randomization.
        """
        # Randomly offset the vehicle along the road normal.
        offset = np.random.uniform(
            low=[-self.road_offset_lateral, -self.road_offset_angular],
            high=[self.road_offset_lateral, self.road_offset_angular],
        )

        # Compute the road normal vector.
        road_normal_x = math.cos(yaw + math.pi / 2)
        road_normal_y = math.sin(yaw + math.pi / 2)

        # Apply the offset to the vehicle's position.
        x += offset[0] * road_normal_x
        y += offset[0] * road_normal_y
        yaw += offset[1]
        yaw = wrap_angle(yaw)  # Wrap yaw to [-pi, pi]

        return x, y, yaw

    def randomize_ego_position(self):
        """
        Randomize the ego vehicle's position within the track bounds.
        This is useful for domain randomization.
        """
        if self.track is None:
            return

        # Randomly select a point on the track to place the ego vehicle.
        track_index = random.choice(range(len(self.track)))
        # We recompute the angle because the angles provided in the track do not seem to be consistent over the track.
        track_loc = self.track[track_index]
        track_loc_next = self.track[(track_index + 1) % len(self.track)]
        angle, pos_x, pos_y = track_loc[1:4]
        # Angle of the track at this point.
        real_angle = np.arctan2(track_loc_next[3] - pos_y, track_loc_next[2] - pos_x)
        # Convert to Box2D angle (perpendicular to track direction).
        real_angle = (real_angle - np.pi / 2)
        yaw = wrap_angle(real_angle)  # Wrap yaw to [-pi, pi]
        if self.randomize_road_position:
            pos_x, pos_y, yaw = self.do_randomize_road_position(pos_x, pos_y, yaw)
        self.start_dict["ego"] = {"x": pos_x, "y": pos_y, "z": 0, "angle": yaw}
        print(f"Randomized ego start position to {self.start_dict['ego']}")

    def set_ego_position_to_map_start(self, start_track_frac=DEFAULT_START_TRACK_FRAC, range=DEFAULT_START_RANGE):
        """
        Set the ego vehicle's position to the map start position.
        """
        if self.track is None:
            return

        track_index = (round(len(self.track) * start_track_frac + random.uniform(-range, range))) % len(self.track)

        # We recompute the angle because the angles provided in the track do not seem to be consistent over the track.
        track_loc = self.track[track_index]
        track_loc_next = self.track[(track_index + 1) % len(self.track)]
        angle, pos_x, pos_y = track_loc[1:4]
        # Angle of the track at this point.
        real_angle = np.arctan2(track_loc_next[3] - pos_y, track_loc_next[2] - pos_x)
        # Convert to Box2D angle (perpendicular to track direction).
        real_angle = (real_angle - np.pi / 2)
        yaw = wrap_angle(real_angle)  # Wrap yaw to [-pi, pi]
        if self.randomize_road_position:
            pos_x, pos_y, yaw = self.do_randomize_road_position(pos_x, pos_y, yaw)
        self.start_dict["ego"] = {"x": pos_x, "y": pos_y, "z": 0, "angle": yaw}
        print(f"Set ego start position to {self.start_dict['ego']}")

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        # TODO(jon.decastro) attempt to remove this, and see if we don't need to reconstruct each reset.
        self.world.contactListener_bug_workaround = FrictionDetector(
            self, self.lap_complete_percent
        )
        self.world.contactListener = self.world.contactListener_bug_workaround
        self.reward = np.zeros(self.num_agents)
        self.prev_reward = np.zeros(self.num_agents)
        self.tile_visited_count = [0] * self.num_agents
        self.t = 0.0
        self.new_lap = False
        self.road_poly = []
        self.is_within_range = False
        self.x_ddm = None

        self.ego_has_passed_ado_count = 0
        self.ego_has_already_passed_ado = False

        self.ado_trigger_radius = random.uniform(
            self.ado_trigger_radius_lb, self.ado_trigger_radius_ub
        )

        if self.domain_randomize:
            randomize = True
            if isinstance(options, dict):
                if "randomize" in options:
                    randomize = options["randomize"]

            self._reinit_colors(randomize)

        # Initialize the ado vehicle ahead of the ego vehicle.
        self.car_order = {idx: i for idx, i in enumerate(reversed(range(self.num_agents)))}

        if not self.track_csv:
            while True:
                success = self._create_random_track()
                if success:
                    break
                if self.verbose:
                    print("Track generation failed.  Retrying with another random track...")
        else:
            success = self._create_track_from_csv()
            print(f"Loaded track from csv of length {len(self.track)}")
            assert success

        (angle, pos_x, pos_y) = self.track[0][1:4]

        if self.randomize_ego_start:
            self.randomize_ego_position()
        else:
            self.set_ego_position_to_map_start()

        self.randomize_ego_position()
        self.reset_ado_positions()
        for car_id in range(self.num_agents):

            if self.start_dict is None:
                # Specify line and lateral separation between cars
                lateral_spacing = LATERAL_SPACING

                # index into positions using modulo and pairs
                line_number = math.floor(self.car_order[car_id]) * -LINE_SPACING  # Starts at 0
                side = (2 * (self.car_order[car_id] % 2)) - 1  # either {-1, 1}

                # Compute angle based off of track index for car
                angle = self.track[line_number][1]

                # Compute offset angle (normal to angle of track)
                norm_theta = angle - np.pi / 2

                # Compute offsets from position of original starting line
                new_x = self.track[line_number][2] + (lateral_spacing * np.sin(norm_theta) * side)
                new_y = self.track[line_number][3] + (lateral_spacing * np.cos(norm_theta) * side)
            else:
                key = "ego" if car_id == self.ego_index else self.ado_names[car_id - 1]
                new_x = self.start_dict[key]["x"]
                new_y = self.start_dict[key]["y"]
                if "angle" not in self.start_dict[key]:
                    quaternion = self.start_dict[key]["quaternion"]
                    rot = Rotation.from_quat(quaternion)
                    angle = rot.as_euler("xyz")[2] - np.pi / 2
                else:
                    angle = self.start_dict[key]["angle"]

            # Display spawn locations of cars.
            print(f"Spawning car {car_id} at ({new_x:.0f}, {new_y:.0f}) with orientation {angle}")

            # Create car at location with given angle
            if car_id == self.ego_index:
                self.cars[car_id] = EpicCar(
                    self.world, angle, new_x, new_y, sim_discretization_time=self.step_time
                )
                self.step_size = self.cars[car_id].params["sim_discretization_time"]
            else:
                self.cars[car_id] = Car(self.world, angle, new_x, new_y)
            self.cars[car_id].hull.color = CAR_COLORS[car_id % len(CAR_COLORS)]

            # This will be used to identify the car that touches a particular tile.
            for wheel in self.cars[car_id].wheels:
                wheel.car_id = car_id

        self.tile_visited_count = [0] * self.num_agents
        self.on_same_tile_count = [0] * self.num_agents
        self.episode_tiles_visited = round(self.pct_track_to_visit * len(self.track)) if self.pct_track_to_visit is not None else len(self.track)

        self.prev_ego_position = np.array([self.ego_car.hull.position[0], self.ego_car.hull.position[1]])
        self.ego_velocity = np.array([0.0, 0.0])

        # To debug and view starting track, run:
        # self._render('human')
        # import time
        # time.sleep(2)
        if self.render_mode == "human":
            self.render()
        return self.step(None)[0], {}

    def _apply_actions(self, ego_action, ado_action):
        """Apply the ego and ado actions to their respective cars."""
        if ego_action is not None:
            self.ego_car.steer(-ego_action[STEER_INDEX])
            self.ego_car.gas(ego_action[THROTTLE_INDEX])
            self.ego_car.brake(ego_action[BRAKE_INDEX])

        if ado_action is not None:
            for i in range(self.num_ados):
                if ado_action[i] is None:
                    continue
                self.ado_car(i).steer(-ado_action[i][STEER_INDEX])
                self.ado_car(i).gas(ado_action[i][THROTTLE_INDEX])
                self.ado_car(i).brake(ado_action[i][BRAKE_INDEX])

    def select_closest_ados(self, car_states):
        """Select the max_num_ados closest ados to the ego car."""
        if self.max_num_ados is None:
            return list(range(1, self.num_ados + 1))
        if self.num_ados <= self.max_num_ados:
            return list(range(1, self.num_ados + 1))

        distances = []
        for i in range(self.num_ados):
            ado_position = car_states[i]["position"]
            ego_position = car_states[self.ego_index]["position"]
            distance = np.linalg.norm(np.array(ado_position) - np.array(ego_position))
            distances.append((distance, i))

        distances.sort()
        # +1 to convert from ado index to car_id
        closest_ados = [i + 1 for _, i in distances[: self.max_num_ados]]
        return closest_ados

    def update_state_features(self, car_states):
        """Update state features for all cars.

        Convention for ego-relative quantities is x-forward, y-left, z-up (right-hand rule).

        NOTE: Since this is used by the ua_shared_controller node, all functions should be self-contained
        and not depend on any class variables except those that are constants or configuration parameters.

        Args:
            car_states: A dictionary of car states for all cars in the environment.
        Returns:
            A tuple containing:
                - pixels_state_vec: A list of dictionaries containing pixel observations and state vectors for each car
                - states: A list of state vectors for each car
        """
        if self.verbose:
            print(f" *** update_state_features: car_states: {car_states}")
        closest_ados = self.select_closest_ados(car_states)
        if self.verbose:
            print(f" *** update_state_features: closest_ados: {closest_ados}")

        track_xy = np.array(self.track)[:, 2:4]
        left_xy = np.array(self.track_left)
        right_xy = np.array(self.track_right)

        states = []
        pixels_state_vec = []

        ego_position = car_states[self.ego_index]["position"]
        # Convert Box2D angle to normal Euler convention.
        ego_heading = (car_states[self.ego_index]["heading"] + np.pi / 2)
        ego_rot = Rotation.from_euler("z", ego_heading).as_matrix()[0:2, 0:2]

        for i, (car_id, state) in enumerate(car_states.items()):
            if i >= self.num_agents:
                break
            distances = np.linalg.norm(track_xy - np.array(state["position"]), axis=1)
            next_tile_idx = np.argmin(distances) + STATE_OBS_SHIFT
            sequence = np.linspace(
                0,
                (int(np.ceil(STATE_OBS_LOOKAHEAD / STATE_OBS_STEP_SIZE)) - 1)
                * STATE_OBS_STEP_SIZE,
                int(np.ceil(STATE_OBS_LOOKAHEAD / STATE_OBS_STEP_SIZE)),
                dtype=np.int32,
            )
            obs_tile_center_ids = (sequence + next_tile_idx) % len(track_xy)

            self.state_obs_features[i] = np.stack(
                [
                    track_xy[obs_tile_center_ids, :],
                    left_xy[obs_tile_center_ids, :],
                    right_xy[obs_tile_center_ids, :],
                ]
            )

            ego_angle_xy = (ego_heading, *ego_position)
            if self.pixel_obs:
                pixels_state = {"image": np.zeros((self.num_agents, VIDEO_H, VIDEO_W, 3))}
            else:
                if car_id == self.ego_index:
                    rel_track_xy = self._transform_points(
                        ego_angle_xy, track_xy[obs_tile_center_ids, :]
                    )
                    rel_left_xy = self._transform_points(
                        ego_angle_xy, left_xy[obs_tile_center_ids, :]
                    )
                    rel_right_xy = self._transform_points(
                        ego_angle_xy, right_xy[obs_tile_center_ids, :]
                    )
                    rel_obs_features = np.stack([rel_track_xy, rel_left_xy, rel_right_xy])
                    # Convert to ego body-relative velocities.
                    body_rel_velocity = ego_rot.dot(state["velocity"])
                    if self.use_relative_heading_in_state:
                        curr_tile_idx = np.argmin(distances)
                        track_loc = self.track[curr_tile_idx]
                        track_loc_next = self.track[(curr_tile_idx + 1) % len(self.track)]
                        angle, pos_x, pos_y = track_loc[1:4]
                        # Angle of the track at this point.
                        track_heading_angle = np.arctan2(track_loc_next[3] - pos_y, track_loc_next[2] - pos_x)
                        track_rel_heading = wrap_angle(ego_heading - track_heading_angle)
                        if self.verbose:
                            print(f" *** update_state_features: track_heading_angle: {track_heading_angle}, ego_heading: {ego_heading}, track_rel_heading: {track_rel_heading}")
                        ego_state = np.array([track_rel_heading, *body_rel_velocity])
                    else:
                        ego_state = np.array([ego_heading, *body_rel_velocity])
                    if self.include_ego_position_in_state:
                        ego_state = np.array([*state["position"], heading_rhr, *body_rel_velocity])
                    pixels_state = {
                        "image": np.zeros((1, 1, 1, 3)),
                        "state": np.array((*ego_state, *rel_obs_features.flatten())),
                    }
                else:
                    if car_id not in closest_ados:
                        states.append(np.hstack([np.array(v) for v in state.values()]))
                        continue
                    rel_position = self._transform_points(
                        ego_angle_xy, [state["position"]]
                    ).flatten()
                    rel_angle = wrap_angle(
                        state["heading"] - car_states[self.ego_index]["heading"]
                    )
                    rel_velocity = ego_rot.dot(
                        np.array(state["velocity"])
                        - np.array(car_states[self.ego_index]["velocity"])
                    )
                    rel_state = (*rel_position, rel_angle, *rel_velocity)
                    pixels_state = {
                        "image": np.zeros((1, 1, 1, 3)),
                        "state": np.array(rel_state),
                    }

            states.append(np.hstack([np.array(v) for v in state.values()]))
            pixels_state_vec.append(pixels_state)

        return pixels_state_vec, states

    def _calculate_rewards(self, ego_action, long_relative):
        """Calculate rewards based on actions and state."""
        if ego_action is None:
            return np.zeros(self.num_agents), False, False

        self.reward -= 0.1

        step_reward = self.reward - self.prev_reward
        self.prev_reward = self.reward.copy()

        # Check termination conditions
        terminated = False
        truncated = self.tile_visited_count == len(self.track) or self.new_lap

        for car_id, car in enumerate(self.cars):
            x, y = car.hull.position
            if abs(x) > PLAYFIELD or abs(y) > PLAYFIELD:
                terminated = True
                step_reward[car_id] = -100

        return step_reward, terminated, truncated

    def compute_ego_velocity(self):
        # Note: We need to compute an empirical velocity based on position change, rather than using the Box2D velocity,
        # because the Box2D velocity tends to be erroneous at low speeds and leads to reward hacking.
        if self.prev_ego_position is None:
            self.prev_ego_position = np.array([self.ego_car.hull.position.x, self.ego_car.hull.position.y])
            empirical_velocity = np.array([0.0, 0.0])
        empirical_velocity = np.array([self.ego_car.hull.position.x, self.ego_car.hull.position.y]) - self.prev_ego_position
        empirical_velocity /= (self.step_size + 1e-5)  # Convert to m/s.
        # empirical_velocity *= 50.0  # Scale to match Box2D velocity scale.
        self.prev_ego_position = np.array([self.ego_car.hull.position.x, self.ego_car.hull.position.y])
        self.ego_velocity = empirical_velocity
        print(f"ego empirical velocity: {self.ego_velocity}")

    def step(self, ego_action: Union[np.ndarray, int]):
        """Execute one environment step."""
        # Get and apply ado action
        ado_actions = [None] * (self.num_ados)
        for i in range(self.num_ados):
            ado_action = self.step_ado(i)
            ado_actions[i] = ado_action["action"]

        # Apply actions to both vehicles
        self._apply_actions(ego_action, ado_actions)

        # Step physics
        for i, (car, a) in enumerate(zip(self.cars, [ego_action, *ado_actions])):
            print(f"Car {i} action: {a}")
            if i == self.ego_index:
                car.apply_controls(a)
                car.step()
            else:
                car.step(self.step_time)
            if self.verbose:
                print(f"Car {car.hull.position} action: {a}, speed: {car.hull.linearVelocity}")
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        self.compute_ego_velocity()  # Update empirical ego velocity post-world-step.

        # Update state features and get observations
        # Wrap angles to [-pi, pi]
        angles = {car_id: wrap_angle(car.hull.angle) for car_id, car in enumerate(self.cars)}
        velocities = {car_id: car.hull.linearVelocity for car_id, car in enumerate(self.cars)}
        car_states = {
            car_id: {
                "position": car.hull.position,
                "heading": angles[car_id],
                "velocity": velocities[car_id],
            }
            for car_id, car in enumerate(self.cars)
        }
        car_states[self.ego_index]["velocity"] = self.ego_velocity  # Use empirical velocity for ego car, since the Box2D velocity is nonsensical, as it simply wraps our EPIC physics engine.
        if self.verbose:
            print(f" *** Car states: {car_states}")
        pixels_state_vec, states = self.update_state_features(car_states)

        # Do not penalize fuel spent
        for car in self.cars:
            car.fuel_spent = 0.0

        # Calculate ego position relative to ado
        distances = []
        for i in range(self.num_ados):
            ado_state = [self.ado_car(i).hull.angle, *self.ado_car(i).hull.position]
            distances.append(self._transform_points(ado_state, [self.ego_car.hull.position]))
        # TODO(jon): nearest ado could be BEHIND the ego.
        index = np.argmin([d[0][0] for d in distances])
        ego_relative_to_ado = distances[index]
        long_relative = ego_relative_to_ado[0][1]

        # Calculate rewards and termination conditions
        step_reward, terminated, truncated = self._calculate_rewards(ego_action, long_relative)

        # Prepare ego observation
        pixels_state_ego = {
            "image": pixels_state_vec[0]["image"],
            "state": np.concatenate([v["state"] for v in pixels_state_vec]),
        }

        # Prepare info dict
        actions = [None] * self.num_agents
        actions[self.ego_index] = ego_action if ego_action is not None else self.init_actions()[0]
        for i, ado_action in enumerate(ado_actions):
            actions[self.ado_indices[i]] = (
                ado_action if ado_action is not None else self.init_actions()[1]
            )
        # TODO Ensure these dimensions agree with the return arg from update_state_features()
        states = states if states is not None else [np.zeros(self.state_length)] * self.num_agents
        info = {"agent_actions": actions, "agent_states": states}

        if self.render_mode == "human":
            self.render()

        return pixels_state_ego, step_reward[0], terminated or truncated, info

    def render(self):
        if self.render_mode is None:
            assert self.spec is not None
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym.make("{self.spec.id}", render_mode="rgb_array")'
            )
            return
        else:
            return self._render(self.render_mode)

    def _render(self, mode="human"):
        if mode == "no_render":
            return np.zeros((self.num_agents, VIDEO_H, VIDEO_W, 3))
        result = []
        for cur_car_id in range(self.num_agents):
            result.append(self._render_window(cur_car_id, mode))

        return np.stack(result, axis=0)

    def debug_render(self):
        self._render_window(0, "human")
        return np.zeros((self.num_agents, VIDEO_H, VIDEO_W, 3))

    def _render_window(self, car_id: int, mode: str):
        assert mode in self.metadata["render_modes"]

        car = self.cars[car_id]
        pygame.font.init()
        if self.screen is None and mode == "human":
            pygame.init()
            pygame.display.init()
            self.screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        if "t" not in self.__dict__:
            return  # reset() not called yet

        self.surf = pygame.Surface((WINDOW_W, WINDOW_H))

        assert car is not None
        # computing transformations
        angle = -car.hull.angle
        # Animating first second zoom.
        zoom = 0.1 * SCALE * max(1 - self.t, 0) + ZOOM * SCALE * min(self.t, 1)
        scroll_x = -(car.hull.position[0]) * zoom
        scroll_y = -(car.hull.position[1]) * zoom
        trans = pygame.math.Vector2((scroll_x, scroll_y)).rotate_rad(angle)
        trans = (WINDOW_W / 2 + trans[0], WINDOW_H / 4 + trans[1])

        self._render_road(zoom, trans, angle)
        for id, car in enumerate(self.cars):
            car.hull.color = (0.0, 0.0, 0.8)  # Set all other car colors to blue
            if id == car_id:  # Ego car
                car.hull.color = (0.8, 0.0, 0.0)  # Set ego car color to red
            car.draw(
                self.surf,
                zoom,
                trans,
                angle,
                mode not in ["state_pixels_list", "state_pixels"],
            )
        # self._render_obs_features(car_id, zoom, trans, angle)
        self.surf = pygame.transform.flip(self.surf, False, True)

        # showing stats
        self._render_indicators(car, WINDOW_W, WINDOW_H)

        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render("%04i" % self.reward[car_id], True, (255, 255, 255), (0, 0, 0))
        text_rect = text.get_rect()
        text_rect.center = (60, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        self.surf.blit(text, text_rect)

        # ego_pos = self.ego_car.hull.position
        # TODO(jon) This is commented out because it is not clear what the ado_pos and RANGE_OVERTAKE is for multiple ados.
        # ado_pos = self.ado_car(i).hull.position
        # ego_ado_distance = np.linalg.norm(ado_pos - ego_pos, axis=0)
        # if ego_ado_distance < RANGE_OVERTAKE:
        #     font = pygame.font.Font(pygame.font.get_default_font(), 42)
        #     text = font.render(self.ego_decision, True, (255, 255, 255), (0, 0, 0))
        #     text_rect = text.get_rect()
        #     text_rect.center = (WINDOW_W - 120, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        #     self.surf.blit(text, text_rect)

        #     font = pygame.font.Font(pygame.font.get_default_font(), 42)
        #     text = font.render(self.ai_acceptance, True, (255, 255, 255), (0, 0, 0))
        #     text_rect = text.get_rect()
        #     text_rect.center = (WINDOW_W - 500, WINDOW_H - WINDOW_H * 2.5 / 40.0)
        #     self.surf.blit(text, text_rect)

        if mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            assert self.screen is not None
            self.screen.fill(0)
            self.screen.blit(self.surf, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return self._create_image_array(self.surf, (VIDEO_W, VIDEO_H))
        elif mode == "state_pixels":
            return self._create_image_array(self.surf, (STATE_W, STATE_H))
        else:
            return self.isopen

    def _render_obs_features(self, car_id, zoom, translation, angle):
        if self.state_obs_features[car_id] is not None:
            self._draw_points(self.state_obs_features[car_id][0], zoom, translation, angle)

    def _render_road(self, zoom, translation, angle):
        bounds = PLAYFIELD
        field = [
            (bounds, bounds),
            (bounds, -bounds),
            (-bounds, -bounds),
            (-bounds, bounds),
        ]

        # draw background
        self._draw_colored_polygon(
            self.surf, field, self.bg_color, zoom, translation, angle, clip=False
        )

        # draw grass patches
        grass = []
        for x in range(-20, 20, 2):
            for y in range(-20, 20, 2):
                grass.append(
                    [
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + 0),
                        (GRASS_DIM * x + 0, GRASS_DIM * y + GRASS_DIM),
                        (GRASS_DIM * x + GRASS_DIM, GRASS_DIM * y + GRASS_DIM),
                    ]
                )
        for poly in grass:
            self._draw_colored_polygon(self.surf, poly, self.grass_color, zoom, translation, angle)

        # draw road
        for poly, color in self.road_poly:
            # converting to pixel coordinates
            poly = [(p[0], p[1]) for p in poly]
            color = [int(c) for c in color]
            self._draw_colored_polygon(self.surf, poly, color, zoom, translation, angle)

    def _render_indicators(self, car, W, H):
        s = W / 40.0
        h = H / 40.0
        color = (0, 0, 0)
        polygon = [(W, H), (W, H - 5 * h), (0, H - 5 * h), (0, H)]
        pygame.draw.polygon(self.surf, color=color, points=polygon)

        def vertical_ind(place, val):
            return [
                (place * s, H - (h + h * val)),
                ((place + 1) * s, H - (h + h * val)),
                ((place + 1) * s, H - h),
                ((place + 0) * s, H - h),
            ]

        def horiz_ind(place, val):
            return [
                ((place + 0) * s, H - 4 * h),
                ((place + val) * s, H - 4 * h),
                ((place + val) * s, H - 2 * h),
                ((place + 0) * s, H - 2 * h),
            ]

        assert car is not None
        true_speed = np.sqrt(
            np.square(car.hull.linearVelocity[0]) + np.square(car.hull.linearVelocity[1])
        )

        # simple wrapper to render if the indicator value is above a threshold
        def render_if_min(value, points, color):
            if abs(value) > 1e-4:
                pygame.draw.polygon(self.surf, points=points, color=color)

        render_if_min(true_speed, vertical_ind(5, 0.02 * true_speed), (255, 255, 255))

        # TODO It's unclear from this code / documentation what the indicators are.  Update documentation and code, accordingly.
        # ABS sensors
        render_if_min(
            car.wheels[0].omega,
            vertical_ind(7, 0.01 * car.wheels[0].omega),
            (0, 0, 255),
        )
        render_if_min(
            car.wheels[1].omega,
            vertical_ind(8, 0.01 * car.wheels[1].omega),
            (0, 0, 255),
        )
        render_if_min(
            car.wheels[2].omega,
            vertical_ind(9, 0.01 * car.wheels[2].omega),
            (51, 0, 255),
        )
        render_if_min(
            car.wheels[3].omega,
            vertical_ind(10, 0.01 * car.wheels[3].omega),
            (51, 0, 255),
        )

        render_if_min(
            car.wheels[0].joint.angle,
            horiz_ind(20, -10.0 * car.wheels[0].joint.angle),
            (0, 255, 0),
        )
        render_if_min(
            car.hull.angularVelocity,
            horiz_ind(30, -0.8 * car.hull.angularVelocity),
            (255, 0, 0),
        )

    def _draw_points(self, points, zoom, translation, angle, radius=5, color=(255, 0, 0)):
        points = [pygame.math.Vector2(*c).rotate_rad(angle) for c in points]
        points = [(c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in points]

        for point in points:
            if (0 <= point[0] < WINDOW_W) and (0 <= point[1] < WINDOW_H):
                gfxdraw.filled_circle(self.surf, int(point[0]), int(point[1]), radius, color)

    def _transform_points(self, state, points_xy, zoom=1):
        """Transform points from world coordinates to car-relative coordinates.
        state: [angle, x, y]
        points_xy: [[x1, y1], [x2, y2], ...]
        zoom: scaling factor (used for rendering)
        returns: [[x1', y1'], [x2', y2'], ...] in car-relative coordinates
        """
        angle = -state[0]
        x = -state[1]
        y = -state[2]
        translation = pygame.math.Vector2((x, y)).rotate_rad(angle)
        points_xy = [pygame.math.Vector2(*c).rotate_rad(angle) for c in points_xy]
        points_xy = [
            (c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in points_xy
        ]
        return np.array(points_xy)

    def _draw_colored_polygon(self, surface, poly, color, zoom, translation, angle, clip=True):
        poly = [pygame.math.Vector2(c).rotate_rad(angle) for c in poly]
        poly = [(c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in poly]
        # This checks if the polygon is out of bounds of the screen, and we skip drawing if so.
        # Instead of calculating exactly if the polygon and screen overlap,
        # we simply check if the polygon is in a larger bounding box whose dimension
        # is greater than the screen by MAX_SHAPE_DIM, which is the maximum
        # diagonal length of an environment object
        if not clip or any(
            (-MAX_SHAPE_DIM <= coord[0] <= WINDOW_W + MAX_SHAPE_DIM)
            and (-MAX_SHAPE_DIM <= coord[1] <= WINDOW_H + MAX_SHAPE_DIM)
            for coord in poly
        ):
            gfxdraw.aapolygon(self.surf, poly, color)
            gfxdraw.filled_polygon(self.surf, poly, color)

    def _create_image_array(self, screen, size):
        scaled_screen = pygame.transform.smoothscale(screen, size)
        return np.transpose(np.array(pygame.surfarray.pixels3d(scaled_screen)), axes=(1, 0, 2))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            self.isopen = False
            pygame.quit()

    def get_action_meanings(self):
        return self.actions
