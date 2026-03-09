# Shared control racing environment using Box2D car dynamics and physics engine.

from copy import deepcopy
import math
import pandas as pd
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
    raise DependencyNotInstalled("Box2D is not installed, run `pip install gymnasium[box2d]`") from e

try:
    # As pygame is necessary for using the environment (reset and step) even without a render mode
    #   therefore, pygame is a necessary import for the environment.
    import pygame
    from pygame import gfxdraw
except ImportError as e:
    raise DependencyNotInstalled("pygame is not installed, run `pip install gymnasium[box2d]`") from e

from util import normalize_track_columns

DEVICE = "cuda:0"

VID_SCALE = 100  # Set to tiny scale, to effectively skip videos when using proprioceptive observations.
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
ZOOM = 0.5  # Camera zoom
ZOOM_FOLLOW = True  # Set to False for fixed view (don't use zoom)

TRACK_DETAIL_STEP = 21 / SCALE
TRACK_TURN_RATE = 0.31
TRACK_WIDTH = 40 / SCALE
BORDER = 8 / SCALE
BORDER_MIN_COUNT = 4
GRASS_DIM = PLAYFIELD / 20.0
MAX_SHAPE_DIM = max(GRASS_DIM, TRACK_WIDTH, TRACK_DETAIL_STEP) * math.sqrt(2) * ZOOM * SCALE

ADO_STATE_OBS_LOOKAHEAD = 20
ADO_STATE_OBS_SHIFT = 2  # Offset of tiles in state observation. 0 = starting with the tile closest to the car
STATE_OBS_LOOKAHEAD = 10  # Number of tiles to look ahead in the state observation
STATE_OBS_SHIFT = 4  # Offset of tiles in state observation. 0 = starting with the tile closest to the car

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

# Penalizing backwards driving
BACKWARD_THRESHOLD = np.pi / 2
K_BACKWARD = 0  # Penalty weight: backwards_penalty = K_BACKWARD * angle_diff  (if angle_diff > BACKWARD_THRESHOLD)

# Overtake / Stay longitudinal penalty activation bias and scalar
BIAS_LONGITUDINAL_OVERTAKE = 1.0
K_OVERTAKE = 2.0
RANGE_OVERTAKE = 40
K_AI_PENALTY = 1.0

# Pure-pursuit params.
K_LOOKAHEAD = 3  # segments ahead
K_SPEED = 20
K_PROP = 0.1
K_WB = 2.9  # wheel base of vehicle

START_LOCATIONS = {
    "straightaway_start": {
        "ego_x": -634.9896240234375,
        "ego_y": -353.8929748535156,
        "ego_z": -3.844970226287842,
        "ego_quaternion": [-0.005201782812592882, -0.01658126567856129, 0.7933157349075446, 0.6085623614981213],
        # 'ego_heading': 0.1296370055848044,  # Heading is the track.csv angle, which is more aligned with box2d's road
        "ado_x": -630.0101928710938,
        "ado_y": -352.9974060058594,
        "ado_z": -4.035552501678467,
        "ado_quaternion": [-0.00660241206413546, -0.013976467610844938, 0.8188548458859308, 0.5737924780590863],
        # 'ado_heading': 0.1296370055848044
    },
    "hairpin_start": {
        "ego_x": -856.32765198,
        "ego_y": -512.35585531,
        "ego_z": -10.867579460144043,
        "ego_quaternion": [-0.016388430669336398, 0.002943737405695212, -0.4871519672306504, 0.8731584704814986],
        # 'ego_heading': 3.6765770308565564,
        "ado_x": -852.50056763,
        "ado_y": -526.93034668,
        "ado_z": -10.574911117553711,
        "ado_quaternion": [-0.015933495891810597, 0.005075089211893106, -0.49626364306659493, 0.8680108085435505],
        # 'ado_heading': 3.852638385163461
    },
}


def proportional_control(current_speed):
    a = K_PROP * (K_SPEED - current_speed)
    return a


def pure_pursuit(targets, state):
    assert K_LOOKAHEAD < len(targets)
    target = targets[K_LOOKAHEAD]
    tx = target[0]
    ty = target[1]
    yaw = state[2]
    alpha = 0.4 * (math.atan2(tx, -ty) - yaw)
    delta = math.atan2(2.0 * K_WB * math.sin(alpha) / np.linalg.norm(target), 1.0)
    # print(f"  target {target}, yaw {yaw}, alpha {alpha}, delta {delta}")
    return delta


class FrictionDetector(contactListener):
    def __init__(self, env, lap_complete_percent):
        contactListener.__init__(self)
        self.env = env
        self.lap_complete_percent = lap_complete_percent

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

                # The reward is dampened on tiles that have been visited already.
                past_visitors = sum(tile.road_visited) - 1
                reward_factor = 1 - (past_visitors / self.env.num_agents)
                self.env.reward[obj.car_id] += reward_factor * 1000.0 / len(self.env.track)
                # print(f"  car {obj.car_id} incrementing reward by {reward_factor * 1000.0 / len(self.env.track)}")

                # Lap is considered completed if enough % of the track was covered
                if (
                    tile.idx == 0
                    and self.env.tile_visited_count[obj.car_id] / len(self.env.track) > self.lap_complete_percent
                ):
                    self.env.new_lap[obj.car_id] = True
        else:
            obj.tiles.remove(tile)


class Box2dSharedRacing(gym.Env, EzPickle):
    """
    Shared control racing environment using Box2D car dynamics and physics engine.

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

    Attributes:
        metadata (dict): The metadata for the environment.
    """

    metadata = {
        "render_modes": ["rgb_array", "state_pixels", "no_render", "human", "bev"],
        "render_fps": FPS,
    }

    def __init__(
        self,
        # ado_agent_path: str,
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
        self.lap_complete_percent = lap_complete_percent  # TODO: Vectorize this per the number of agents?
        self._init_colors()

        self.contactListener_keepref = FrictionDetector(self, self.lap_complete_percent)
        self.world = Box2D.b2World((0, 0), contactListener=self.contactListener_keepref)
        self.screen: Optional[pygame.Surface] = None

        # self.viewer = [None] * num_agents
        self.surf = None
        self.clock = None
        self.isopen = True
        self.invisible_state_window = None
        self.invisible_video_window = None
        self.road = None
        # self.car: Optional[Car] = None
        self.cars = [None] * num_agents
        self.car_order = None  # Determines starting positions of cars
        self.reward = np.zeros(num_agents)
        self.prev_reward = np.zeros(num_agents)
        self.tile_visited_count = [0] * num_agents
        self.verbose = verbose
        self.new_lap = False
        self.ado_rails = ado_rails
        self.ado_names = ado_names
        self.ado_trigger_groups = ado_trigger_groups
        self.fd_tile = fixtureDef(shape=polygonShape(vertices=[(0, 0), (1, 0), (1, -1), (0, -1)]))
        self.state_obs_features = [None] * num_agents  # Storage for road features that are part of the state.
        self.track = None
        self.track_left = None
        self.track_right = None
        self.ado_state = None
        self.ego_decision = None
        self.ai_decision = None
        self.x_ddm = None
        # self.ai_action_at_onset = None
        self.ai_acceptance = None
        self.is_within_range = False
        self.track_csv = track_csv
        self.track_df = pd.read_csv(track_csv)
        self.road_poly = []
        self.ego_has_passed_ado_count = 0
        self.ego_has_already_passed_ado = False

        self.ego_index = 0
        self.ado_index = 1

        self.use_pure_pursuit = config.use_pure_pursuit if config is not None else False
        self.use_race_line = config.use_race_line if config is not None else False

        # This will throw a warning in tests/envs/test_envs in utils/env_checker.py as the space is not symmetric
        #   or normalised however this is not possible here so ignore

        # We're training only for the ego vehicle here; its observation space should include the ado's state.
        if self.continuous:
            action_lb = np.array([-1, 0, 0]).astype(np.float64)  # ai steer, gas, brake, human steer, gas, brake
            action_ub = np.array([+1, +1, +1]).astype(np.float64)  # ai steer, gas, brake, human steer, gas, brake
            self.action_space = spaces.Box(action_lb, action_ub)
            self.actions = ["STEER_A", "GAS_A", "BRAKE_A"]
        else:
            raise TypeError("Must be a continuous action space!")

        self.render_mode = "state_pixels" if render_mode is not "no_render" else render_mode
        if self.pixel_obs:
            self.observation_space = spaces.Dict(
                image=spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8),
            )
        else:
            self.render_mode = "rgb_array" if render_mode is None else render_mode  # Makes train mode more efficient.
            # Base observation includes ego state + track tiles + other agents' states
            obs_lb = np.array(
                (
                    *[-math.pi] * num_agents,
                    *[-PLAYFIELD] * 2 * num_agents,
                    *[-PLAYFIELD] * (2 * STATE_OBS_LOOKAHEAD) * 3,
                )
            )
            # Add the bounds for any additional agents besides the ego agent.
            for agent in range(num_agents-1):
                obs_lb = np.append(obs_lb, np.array(
                    (
                        *[-PLAYFIELD] * 2,  # x, y position
                        *[-math.pi],        # heading
                        *[0.0] * 2,         # velocities
                    )
                ))
            obs_ub = np.array(
                (*[math.pi] * num_agents, *[PLAYFIELD] * 2 * num_agents, *[PLAYFIELD] * (2 * STATE_OBS_LOOKAHEAD) * 3)
            )
            # Add the bounds for any additional agents besides the ego agent.
            for agent in range(num_agents-1):
                obs_ub = np.append(obs_ub, np.array(
                    (
                        *[PLAYFIELD] * 2,   # x, y position
                        *[math.pi],         # heading
                        *[100.0] * 2,       # velocities (match epic_dynamics)
                    )
                ))
            self.observation_space = spaces.Dict(
                image=spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8),
                state=spaces.Box(low=obs_lb, high=obs_ub),
            )
        # Note: The above observation spaces should not account for the actions of other agents. These will be
        # automatically appended to the observation space in the wrapper.

        # Ado action / observation spaces should be consistent with the trained policy.
        action_lb = np.array([-1, 0, 0]).astype(np.float32)  # steer, gas, brake
        action_ub = np.array([+1, +1, +1]).astype(np.float32)  # steer, gas, brake
        # self.ado_action_space = spaces.Box(action_lb, action_ub)
        self.ado_action_space = spaces.Box(action_lb, action_ub)
        obs_lb = np.array((-math.pi, *[-PLAYFIELD] * 2, *[-PLAYFIELD] * (2 * ADO_STATE_OBS_LOOKAHEAD) * 3))
        obs_ub = np.array((math.pi, *[PLAYFIELD] * 2, *[PLAYFIELD] * (2 * ADO_STATE_OBS_LOOKAHEAD) * 3))
        self.ado_observation_space = spaces.Dict(
            image=spaces.Box(low=0, high=255, shape=(STATE_H, STATE_W, 3), dtype=np.uint8),
            state=spaces.Box(low=obs_lb, high=obs_ub),
        )

        # Since the ado had been trained in a separate (non-reactive) environment, we want to ensure we use that same env here.
        # config_in = deepcopy(config)  # Prevent shadowing of the config, as it will be mutated in DreamerPolicy.
        # self.ado_env = make_env(config_in)
        # acts = self.ado_env.action_space
        # config_in.num_actions = acts.n if hasattr(acts, "n") else acts.shape[0]
        # self.ado_agent = DreamerPolicy(self.ado_observation_space, self.ado_action_space, config_in,).to(DEVICE)
        # assert pathlib.Path(ado_agent_path).expanduser().exists()
        # # self.ado_agent = torch.load(ado_agent_path).to(DEVICE)
        # self.ado_agent.load_state_dict(torch.load(ado_agent_path))
        # self.ado_agent.requires_grad_(requires_grad=False)
        # # self.ado_agent._should_pretrain._once = False
        # self.ado_agent = functools.partial(self.ado_agent)

        self.ado_agent = dreamer_policy

        # if self.track_csv:
        #     success = self._create_track_from_csv()
        #     assert success

        self.start_dict = start_position
        if self.start_dict is not None:
            if self.start_dict not in START_LOCATIONS:
                raise AttributeError(
                    f"Given start position {self.start_dict} does not exist "
                    f"in known start positions. Ensure that your start position has associated "
                    f"data in human_ai_racing START_LOCATIONS dict."
                )
            self.start_dict = START_LOCATIONS[self.start_dict]

    @property
    def ego_car(self):
        assert self.cars[self.ego_index] is not None
        return self.cars[self.ego_index]


    def init_actions(self):
        return np.zeros(len(self.action_space.low)), np.zeros(len(self.action_space.low))

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

        # For town format, compute velocity from components BEFORE normalization
        # (path_vx and path_vy need to be combined before renaming)
        # Note: since we never use df['path_vx'] and df['path_vy'] directly and only use the normalized 'refline/v',
        # this should not cause any issues with column renaming downstream.
        if 'path_vx' in df.columns and 'path_vy' in df.columns:
            df['path_vx'] = np.sqrt(df['path_vx']**2 + df['path_vy']**2)
            # Drop path_vy as it's now incorporated into path_vx
            df = df.drop(columns=['path_vy'])

        # Normalize column names to handle different CSV formats using shared utility
        df, _ = normalize_track_columns(df, target_format='slash')

        # Set default refline velocity if not available
        if 'refline/v' not in df.columns:
            df['refline/v'] = 0.0

        track = np.array(
            [
                df.loc[:, "inner_edge/x"],
                df.loc[:, "inner_edge/y"],
                df.loc[:, "outer_edge/x"],
                df.loc[:, "outer_edge/y"],
                df.loc[:, "refline/x"],
                df.loc[:, "refline/y"],
                df.loc[:, "refline/v"],
            ]
        )
        track = track[:, :-1].T  # The final entry is just a repeat of the first one.

        # Red-white border on hard turns
        border = [False] * len(track)

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
            alpha = 0.0  # unused.
            if self.use_race_line:
                ref_line1 = track[i, 4:5]
                ref_line2 = track[i - 1, 4:5]
            else:
                ref_line1 = (road1_l + road1_r) / 2
                ref_line2 = (road2_l + road2_r) / 2
            beta = math.atan((ref_line1[1] - ref_line2[1]) / (ref_line1[0] - ref_line2[0])) + math.pi / 2
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

    def _create_track(self):
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
            pass_through_start = track[i][0] > self.start_alpha and track[i - 1][0] <= self.start_alpha
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

    def get_ado_obs(self):
        # Update state features
        track_xy = np.array(self.track)[:, 2:4]
        left_xy = np.array(self.track_left)
        right_xy = np.array(self.track_right)

        distances = np.linalg.norm(track_xy - self.ado_car(0).hull.position, axis=1)
        next_tile_idx = np.argmin(distances) + ADO_STATE_OBS_SHIFT
        obs_tile_center_ids = (np.array(range(ADO_STATE_OBS_LOOKAHEAD)) + next_tile_idx) % len(track_xy)

        rel_track_xy = self._transform_points(self.ado_car(0), track_xy[obs_tile_center_ids, :])
        rel_left_xy = self._transform_points(self.ado_car(0), left_xy[obs_tile_center_ids, :])
        rel_right_xy = self._transform_points(self.ado_car(0), right_xy[obs_tile_center_ids, :])
        rel_obs_features = np.stack([rel_track_xy, rel_left_xy, rel_right_xy])

        pixels_state = {
            "image": np.zeros((self.num_agents, VIDEO_H, VIDEO_W, 3)),
            "state": np.array(
                (
                    self.ado_car(0).hull.angle,
                    *self.ado_car(0).hull.position,
                    *rel_obs_features.flatten(),
                )
            ),
            "reward": 0,
            "is_terminal": False,
            "is_first": False,
        }
        return [pixels_state]

    def pure_pursuit(self, obs):
        # Assumes shared_control_racing obs is [angle, position, flattened_features]
        state = [*obs["state"][0][1:3], obs["state"][0][0]]
        rel_features = obs["state"][0][3:].reshape(3, -1, 2)  # Reshape flattened_features to a 3 x 2 x N array
        targets = rel_features[0]
        delta = pure_pursuit(targets, state)
        speed = np.sqrt(np.square(self.ado_car(0).hull.linearVelocity[0]) + np.square(self.ado_car(0).hull.linearVelocity[1]))
        a = proportional_control(speed)
        action = {"action": torch.tensor([[np.clip(delta, -1, 1), np.clip(a, 0, 1), np.clip(-a, 0, 1)]])}
        return action

    def step_ado(self):
        # Initialize or unpack simulation state.
        # Step agent.
        obs = self.get_ado_obs()
        done = np.array([False])
        obs = {k: np.stack([obs[0][k]]) for k in obs[0]}
        if self.use_pure_pursuit:
            action = self.pure_pursuit(obs)
        else:
            action, self.ado_state = self.ado_agent(obs, done, self.ado_state)
        if isinstance(action, dict):
            action = {k: np.array(action[k][0].detach().cpu()) for k in action}
        else:
            action = np.array(action)
        return action

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)
        self._destroy()
        # TODO(jon.decastro) attempt to remove this, and see if we don't need to reconstruct each reset.
        self.world.contactListener_bug_workaround = FrictionDetector(self, self.lap_complete_percent)
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
                success = self._create_track()
                if success:
                    break
                if self.verbose:
                    print("retry to generate track (normal if there are not many" "instances of this message)")
        else:
            success = self._create_track_from_csv()
            assert success

        # self.car = Car(self.world, *self.track[0][1:4])
        (angle, pos_x, pos_y) = self.track[0][1:4]
        car_width = car_dynamics.SIZE * (
            car_dynamics.WHEEL_W * 2 + (car_dynamics.WHEELPOS[1][0] - car_dynamics.WHEELPOS[1][0])
        )
        for car_id in range(self.num_agents):

            if self.start_dict is None:
                # Specify line and lateral separation between cars
                lateral_spacing = LATERAL_SPACING

                # index into positions using modulo and pairs
                line_number = math.floor(self.car_order[car_id]) * -LINE_SPACING  # Starts at 0
                side = (2 * (self.car_order[car_id] % 2)) - 1  # either {-1, 1}

                # Compute angle based off of track index for car
                angle = self.track[line_number][1]
                #
                # # Compute offset angle (normal to angle of track)
                norm_theta = angle - np.pi / 2
                #
                # # Compute offsets from position of original starting line
                new_x = self.track[line_number][2] + (lateral_spacing * np.sin(norm_theta) * side)
                new_y = self.track[line_number][3] + (lateral_spacing * np.cos(norm_theta) * side)
                print(f"Spawning car {car_id} at ({new_x:.0f}, {new_y:.0f}) with orientation {angle}")
            else:
                key = "ego" if car_id == 0 else "ado"
                new_x = self.start_dict[f"{key}_x"]
                new_y = self.start_dict[f"{key}_y"]
                # angle = self.start_dict[f'{key}_heading']
                quaternion = self.start_dict[f"{key}_quaternion"]
                rot = Rotation.from_quat(quaternion)
                angle = rot.as_euler("xyz")[2] - np.pi / 2

            # Display spawn locations of cars.
            # print(f"Spawning car {car_id} at ({new_x:.0f}, {new_y:.0f}) with orientation {angle}")

            # Create car at location with given angle
            self.cars[car_id] = Car(self.world, angle, new_x, new_y)
            self.cars[car_id].hull.color = CAR_COLORS[car_id % len(CAR_COLORS)]

            # This will be used to identify the car that touches a particular tile.
            for wheel in self.cars[car_id].wheels:
                wheel.car_id = car_id

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
            self.ego_car.steer(-ego_action[0])
            self.ego_car.gas(ego_action[1])
            self.ego_car.brake(ego_action[2])

        if ado_action is not None:
            self.ado_car(0).steer(-ado_action[0])
            self.ado_car(0).gas(ado_action[1])
            self.ado_car(0).brake(ado_action[2])

    def _update_state_features(self):
        """Update state features for all cars."""
        track_xy = np.array(self.track)[:, 2:4]
        left_xy = np.array(self.track_left)
        right_xy = np.array(self.track_right)

        states = []
        pixels_state_vec = []
        render = self._render(self.render_mode)

        for car_id, car in enumerate(self.cars):
            car.fuel_spent = 0.0
            distances = np.linalg.norm(track_xy - car.hull.position, axis=1)
            next_tile_idx = np.argmin(distances) + STATE_OBS_SHIFT
            obs_title_center_ids = (np.array(range(STATE_OBS_LOOKAHEAD)) + next_tile_idx) % len(track_xy)

            self.state_obs_features[car_id] = np.stack(
                [
                    track_xy[obs_title_center_ids, :],
                    left_xy[obs_title_center_ids, :],
                    right_xy[obs_title_center_ids, :],
                ]
            )
            rel_track_xy = self._transform_points(car, track_xy[obs_title_center_ids, :])
            rel_left_xy = self._transform_points(car, left_xy[obs_title_center_ids, :])
            rel_right_xy = self._transform_points(car, right_xy[obs_title_center_ids, :])
            rel_obs_features = np.stack([rel_track_xy, rel_left_xy, rel_right_xy])

            if self.pixel_obs:
                pixels_state = {"image": render[car_id]}
            else:
                # Include all agents' base states + track tiles + relative states of other agents
                all_angles = [c.hull.angle for c in self.cars]
                all_positions = [coord for c in self.cars for coord in c.hull.position]
                base_state = (*all_angles, *all_positions, *rel_obs_features.flatten())

                # Add relative states of other agents (for num_agents > 1)
                other_agent_states = []
                for other_car_id, other_car in enumerate(self.cars):
                    if other_car_id != car_id:
                        # Relative position, heading, and velocity of other agent
                        rel_pos = self._transform_points(car, [other_car.hull.position])[0]
                        rel_angle = other_car.hull.angle - car.hull.angle
                        # Note: velocities are approximated as 0 for now (Box2D doesn't directly expose linear velocity)
                        other_agent_states.extend([*rel_pos, rel_angle, 0.0, 0.0])

                pixels_state = {
                    "image": render,
                    "state": np.array((*base_state, *other_agent_states)),
                }

            states.append((*car.hull.position, car.hull.angle, 0.0, 0.0))
            # Note: velocities are approximated as 0 for now (Box2D doesn't directly expose linear velocity)
            pixels_state_vec.append(pixels_state)

        pixels_state_vec = {
            k: np.stack([v1 for d in pixels_state_vec for k1, v1 in d.items() if k == k1]) for k in pixels_state_vec[0]
        }

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

    def step(self, ego_action: Union[np.ndarray, int]):
        """Execute one environment step."""
        # Get and apply ado action
        ado_action = None
        if ego_action is not None:
            # Only apply the ado action when there is an ego action
            # This is in place to ensure that the ado action isn't automatically applied when the environment is reset.
            ado_action = self.step_ado()
            ado_action = ado_action["action"]

        # Apply actions to both vehicles
        self._apply_actions(ego_action, ado_action)

        # Step physics
        for car in self.cars:
            car.step(1.0 / FPS)
        self.world.Step(1.0 / FPS, 6 * 30, 2 * 30)
        self.t += 1.0 / FPS

        # Update state features and get observations
        pixels_state_vec, states = self._update_state_features()

        # Calculate ego position relative to ado
        ego_relative_to_ado = self._transform_points(self.ado_car(0), [self.ego_car.hull.position])
        long_relative = ego_relative_to_ado[0][1]

        # Calculate rewards and termination conditions
        step_reward, terminated, truncated = self._calculate_rewards(ego_action, long_relative)
        print(f"  step_reward {step_reward}")

        # Prepare ego observation
        # Note: Other agent information is already included in pixels_state_vec["state"][0]
        # by _update_state_features(), so no need to concatenate here
        pixels_state_ego = {
            "image": pixels_state_vec["image"][0],
            "state": pixels_state_vec["state"][0],
        }

        # Prepare info dict
        actions = [None] * self.num_agents
        actions[self.ego_index] = ego_action if ego_action is not None else self.init_actions()[0]
        actions[self.ado_index] = ado_action if ado_action is not None else self.init_actions()[1]
        # Fallback shape is 2 x 5 beacause there are 2 egocentric agents (human, AI) and 5 states each
        states = states if states is not None else [(0.0, 0.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 0.0, 0.0)]

        info = {"agent_actions": actions, "agent_states": states, "reward_components": []}
        # print(info['agent_actions'])
        # print(f"  tile_visited_count {self.tile_visited_count}")

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

        ego_pos = self.ego_car.hull.position
        ado_pos = self.ado_car(0).hull.position
        ego_ado_distance = np.linalg.norm(ado_pos - ego_pos, axis=0)
        if ego_ado_distance < RANGE_OVERTAKE:
            font = pygame.font.Font(pygame.font.get_default_font(), 42)
            text = font.render(self.ego_decision, True, (255, 255, 255), (0, 0, 0))
            text_rect = text.get_rect()
            text_rect.center = (WINDOW_W - 120, WINDOW_H - WINDOW_H * 2.5 / 40.0)
            self.surf.blit(text, text_rect)

            font = pygame.font.Font(pygame.font.get_default_font(), 42)
            text = font.render(self.ai_acceptance, True, (255, 255, 255), (0, 0, 0))
            text_rect = text.get_rect()
            text_rect.center = (WINDOW_W - 500, WINDOW_H - WINDOW_H * 2.5 / 40.0)
            self.surf.blit(text, text_rect)

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
        self._draw_colored_polygon(self.surf, field, self.bg_color, zoom, translation, angle, clip=False)

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
        true_speed = np.sqrt(np.square(car.hull.linearVelocity[0]) + np.square(car.hull.linearVelocity[1]))

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

    def _transform_points(self, state, points, zoom=1):
        """Transform points from world coordinates to car-relative coordinates.
        state: Can be either a car object or a tuple [angle, x, y]
        points: [[x1, y1], [x2, y2], ...]
        zoom: scaling factor (used for rendering)
        returns: [[x1', y1'], [x2', y2'], ...] in car-relative coordinates
        """
        # Handle both car objects and state tuples
        if isinstance(state, tuple) or isinstance(state, list):
            angle = -state[0]
            x = -state[1]
            y = -state[2]
        else:
            # Assume it's a car object
            assert state is not None
            angle = -state.hull.angle
            x = -state.hull.position[0]
            y = -state.hull.position[1]

        translation = pygame.math.Vector2((x, y)).rotate_rad(angle)
        points = [pygame.math.Vector2(*c).rotate_rad(angle) for c in points]
        points = [(c[0] * zoom + translation[0], c[1] * zoom + translation[1]) for c in points]
        return np.array(points)

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

    def reset_ado_rails(self, ado_rails, ado_trigger_groups=None):
        """Reset ADO rails and trigger groups dynamically."""
        self.ado_rails = ado_rails
        self.ado_trigger_groups = ado_trigger_groups

    def ado_car(self, i):
        """Get the i-th ADO car (0-indexed)."""
        if not hasattr(self, 'ado_indices'):
            self.ado_indices = list(range(1, self.num_agents))
        return self.cars[self.ado_indices[i]]

    def get_action_meanings(self):
        return self.actions


if __name__ == "__main__":
    a = np.array([0.0, 0.0, 0.0])

    def register_input():
        global quit, restart
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    a[0] = -1.0
                if event.key == pygame.K_RIGHT:
                    a[0] = +1.0
                if event.key == pygame.K_UP:
                    a[1] = +1.0
                if event.key == pygame.K_DOWN:
                    a[2] = +0.8  # set 1.0 for wheels to block to zero rotation
                if event.key == pygame.K_RETURN:
                    restart = True
                if event.key == pygame.K_ESCAPE:
                    quit = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_LEFT:
                    a[0] = 0
                if event.key == pygame.K_RIGHT:
                    a[0] = 0
                if event.key == pygame.K_UP:
                    a[1] = 0
                if event.key == pygame.K_DOWN:
                    a[2] = 0

            if event.type == pygame.QUIT:
                quit = True

    env = HumanAiRacing(render_mode="human", pixel_obs=False)

    quit = False
    while not quit:
        env.reset()
        total_reward = 0.0
        steps = 0
        restart = False
        while True:
            register_input()
            s, r, terminated, truncated, info = env.step(a)
            total_reward += r
            if steps % 200 == 0 or terminated or truncated:
                print("\naction " + str([f"{x:+0.2f}" for x in a]))
                print(f"step {steps} total_reward {total_reward:+0.2f}")
            steps += 1
            if terminated or truncated or restart or quit:
                break
    env.close()
