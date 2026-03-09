from copy import deepcopy
from collections import OrderedDict
from glob import glob
import gymnasium as gym
from gymnasium import spaces
import math
import numpy as np
import os
import pandas as pd
import pathlib
import random
from scipy.spatial.transform import Rotation as R
import wandb

import envs.box2d_shared_racing
from util import make_absolute_path, wrap_angle

_DIST_THRESHOLD = 15  # Threshold to hit a checkpoint on the track

EGO_INDEX = 0

STEER_INDEX = 0
ACCEL_INDEX = 1

ACTION_INDEX_AI = 0
ACTION_INDEX_HUMAN = 1


def make_env(config, logger, mode, train_eps, eval_eps):
    """Create an environment.

    Parameters:
        config: The configuration.
        logger: The logger.
        mode: The mode.
        train_eps: The training episodes.

    Returns:
        env: The environment.
    """
    # TODO (Jon DeCastro): This function is being saved to incorporate certain elements into
    # training, but it is not actually used anywhere at the moment

    # Load the external human agent policy.
    directory = config.traindir

    ado_action_space = spaces.Box(2)  # TODO: Check that this is correct.

    obs_lb = np.array(
        (
            -math.pi,
            *[-envs.box2d_shared_racing.PLAYFIELD] * 2,
            *[-envs.box2d_shared_racing.PLAYFIELD] * (2 * envs.box2d_shared_racing.ADO_STATE_OBS_LOOKAHEAD) * 3,
        )
    )
    obs_ub = np.array(
        (
            math.pi,
            *[envs.box2d_shared_racing.PLAYFIELD] * 2,
            *[envs.box2d_shared_racing.PLAYFIELD] * (2 * envs.box2d_shared_racing.ADO_STATE_OBS_LOOKAHEAD) * 3,
        )
    )
    ado_observation_space = spaces.Dict(
        image=spaces.Box(
            low=0,
            high=255,
            shape=(envs.box2d_shared_racing.STATE_H, envs.box2d_shared_racing.STATE_W, 3),
            dtype=np.uint8,
        ),
        state=spaces.Box(low=obs_lb, high=obs_ub),
    )
    config = deepcopy(config)
    config.num_actions = ado_action_space.n if hasattr(ado_action_space, "n") else ado_action_space.shape[0]

    logdir = pathlib.Path(config.logdir + "_tmp").expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    tmp_logger = tools.Logger(logdir, config.action_repeat * step)

    # Load the ado-agent
    eval_policy = None
    if not config.use_pure_pursuit:
        config.use_intent = False
        agent = Dreamer(
            ado_observation_space,
            ado_action_space,
            config,
            tmp_logger,
            dataset=None,
        ).to(config.device)
        agent.requires_grad_(requires_grad=False)
        agent.load_state_dict(torch.load(str(make_absolute_path(config.human_ego_agent_paths))))
        agent._should_pretrain._once = False
        eval_policy = functools.partial(agent, training=False)
    config.use_intent = True

    # Load the external human agent policies, and feed those into the active env.
    # labeled_human_agents = make_human_agents(config, task) if config.use_human_inference_reward else {}

    env = Box2dSharedRacingWrapper(
        config=config,
        render_mode=None if mode == "eval" else "no_render",
        track_csv=str(make_absolute_path(config.track_csv)),
        dreamer_policy=eval_policy,
        human_agents=labeled_human_agents,
    )
    renderer = functools.partial(Render, env)
    return env, renderer


def make_human_agents(config, task):
    # TODO: Update me.
    human_config = deepcopy(config)
    env_config_overrides = make_env_config_overrides(human_config)
    human_config.egocentric_agent_names = ["human"]
    human_config.human_agent_paths = []
    human_config.frozen_agent_id = 0  # Ensure the weights of this agent are not updated.
    env_config_overrides_human_only = deepcopy(env_config_overrides)
    env_config_overrides_human_only["sim_config"]["port"] = find_free_port()
    # TODO(jon): If init_carla is true, then we're specifying a port in `make_env_config_overrides` and another one here.
    # We should clean this up and consolidate ports.
    env_config_overrides_human_only["sim_config"]["init_carla"] = True

    human_only_env = HumanAiRacingCarlaWrapper(
        task,
        human_config.action_repeat,
        human_config.size,
        config=human_config,
        track_csv=str(make_absolute_path(human_config.track_csv)),
        config_name=human_config.multicar_racing_config_name,
        config_overrides=env_config_overrides_human_only,
    )

    logdir = pathlib.Path(config.logdir + "_tmp").expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    step = count_steps(human_config.traindir)
    tmp_logger = tools.Logger(logdir, config.action_repeat * step)
    obs_spaces = human_only_env.observation_spaces
    act_spaces = human_only_env.action_spaces
    labeled_human_agents = {}
    for path in config.human_ego_agent_paths:
        human_agents = PopulationDreamer(obs_spaces, act_spaces, human_config, tmp_logger).to(config.device)
        human_agents.requires_grad_(requires_grad=False)
        filename = os.path.join(str(make_absolute_path(path)), "latest_model.pt")
        label = None
        for i, l in human_only_env.get_label_dict().items():
            if l in filename:
                label = i
                break
        assert label is not None
        f"No label found in checkpoint name. Expected {human_only_env.get_label_dict().values()} in the directory name."
        human_agents.load_state_dict(torch.load(filename))
        human_agents._should_pretrain._once = False
        if label in list(labeled_human_agents.keys()):
            raise RuntimeError("Duplicate label found in supplied human agent policy directories.")
        labeled_human_agents[label] = human_agents
    return labeled_human_agents


class Box2dSharedRacingWrapper:
    LOCK = None

    def __init__(
        self,
        config,
        action_repeat=1,
        gray=False,
        noops=0,
        lives="unused",
        sticky=True,
        actions="all",
        length=108000,
        resize="opencv",
        seed=None,
        # human_agent_path=None,
        render_mode=None,
        num_agents=2,
        start_position=None,
        human_agents=None,
        **kwargs,
    ):
        assert lives in ("unused", "discount", "reset"), lives
        assert actions in ("all", "needed"), actions
        assert resize in ("opencv", "pillow"), resize
        if self.LOCK is None:
            import multiprocessing as mp

            mp = mp.get_context("spawn")
            self.LOCK = mp.Lock()
        self._resize = resize
        if self._resize == "opencv":
            import cv2

            self._cv2 = cv2
        if self._resize == "pillow":
            from PIL import Image

            self._image = Image
        self._repeat = action_repeat
        self._gray = gray
        self._noops = noops
        self._lives = lives
        self._sticky = sticky
        self._length = length
        self._random = np.random.RandomState(seed)
        self.config = config
        with self.LOCK:
            # print(human_agent_path)
            self._env = gym.make(
                "Box2dSharedRacing-v0",
                config=config,
                render_mode=render_mode,
                num_agents=num_agents,
                continuous=True,
                track_csv=str(make_absolute_path(config.track_csv)),
                dreamer_policy=human_agents,
                start_position=start_position,
            )
        # assert self._env.unwrapped.get_action_meanings()[0] == "NOOP"

        shape = self._env.observation_space.shape
        self._done = True
        self._step = 0
        self._global_step = None
        self._num_egocentric_agents = len(config.egocentric_agent_names)
        self._ref_ado_csv = None
        self._base_ado_timestep = {}  # Dict: ado_name -> timestep
        self._ref_ado_csv_base = None
        self.ado_slowdown_factor = self._max_ado_slowdown_factor = config.ado_slowdown_factor
        self._ado_speed_factor = 1.0
        self.track_df = str(make_absolute_path(config.track_csv))
        # ADO tracking
        self._ado_names = getattr(config, 'ado_names', ['ado0'])
        self._num_ados = len(self._ado_names)
        self._ado_trigger_groups = None
        self._last_acts_for_reward_shaping = None
        self._running_progress = 0.0

        self._use_action_diff_penalty = config.use_action_diff_penalty
        self._max_steering_penalty = config.max_steering_penalty
        self._steering_penalty_update_rate = config.steering_penalty_update_rate
        self._speed_factor_randomization_prob = config.speed_factor_randomization_prob

        # Curriculum learning flags
        self._headstart_curriculum = getattr(config, 'headstart_curriculum', False)
        self._ado_speed_curriculum = getattr(config, 'ado_speed_curriculum', False)

        # Collision detection flag
        self._use_collision_penalty = getattr(config, 'use_collision_penalty', False)

        # Speed penalty flag
        self._use_speed_penalty = getattr(config, 'use_speed_penalty', False)

        # === Inference-related parameters ===
        self.label = None
        # Currently, we assume that we'll zero out the AI action from the perspective of the human ego driver.
        # This will need to change if we decide to interleave with a reactive human.
        self.frozen_agent_id = getattr(config, 'frozen_agent_id', None)
        self.use_intent = getattr(config, 'use_intent', False)
        self.human_agents = human_agents
        self._human_agent_state = (
            None  # Note: be careful to reset this if an agent's model is switched due to inference.
        )
        self._last_inferred_intent = 0
        self.inferred_intents = None
        self.num_agents = self.base_env.num_agents
        self.action_space = self.base_env.action_space
        self.num_actions = 2  # This is number of *un-allocated* actions.
        self.num_allocated_actions = 3  # steering, throttle, brake (vehicle space)

        self.actions = self.base_env.actions  # These are the *allocated* actions.
        self.continuous = self.base_env.continuous
        self.road_poly = self.base_env.road_poly

        self.vehicle_data = []  # For BEV plotting.

        if hasattr(config, "ado_rails"):
            self._ref_ado_csv_base = str(make_absolute_path(config.ado_rails))
            self._ref_ado_csv_fns = []
            for fn in os.listdir(self._ref_ado_csv_base):
                if fn.endswith(".csv"):
                    self._ref_ado_csv_fns.append(fn)
            self._max_ado_headstart = config.max_ado_headstart
            self._ado_headstart = 0
            self._ado_update_rate = config.ado_update_rate
            if not self._headstart_curriculum:  # If we don't slowly increment, assume we init at max.
                self._ado_headstart = self._max_ado_headstart
            if not self._ado_speed_curriculum:
                self.ado_slowdown_factor = 0.0
            self.reset_ado_rails()

        # If we want to add ado agent metrics in the future, consider something like:
        # 'average_speed': {i: [] for i in range(config.num_agents)}
        self.eval_metrics = {
            "steps_per_lap": [],
            "average_speed": [],
            "max_speed": [],
            "average_intervention_norm": [],
            "max_intervention_norm": [],
            "action_discontinuity": [],
        }
        self._eval_metric_helpers = {
            "lap_start_step": 0,
            "checkpoint_progress": {},
            "current_speed": [],
            "last_checkpoint": "",
            "current_intervention_norm": [],
            "action_discontinuity": [],
            "reward_components": {},
        }
        # Initialize reward_component_names based on configuration
        # Base components (always present)
        self.reward_component_names = [
            "base_reward",
            "speed_bonus",
            "action_smoothness_penalty",
            "bound_penalty",
            "collision_penalty",
            "speed_penalty",
            "left_right_bias_penalty",
        ]

        # Add human_acts_bonus if we have 2 egocentric agents
        # Note: This is just an initial guess; reward_component_names will be updated
        # dynamically during step() to match the actual components generated
        if self._num_egocentric_agents == 2:
            self.reward_component_names.append("human_acts_bonus")

        # Additional metrics for wandb logging
        self.out_of_bound_count = 0
        self.collisions_per_ado_id = {}
        self.incorrect_left_steer_count = 0
        self.incorrect_right_steer_count = 0
        self.incorrect_steering_angle_count = 0

        # Define action space indices.  AI first, if available.
        if self._num_egocentric_agents == 2:
            # For now, we're expecting the AI to be the first agent.
            # TODO(jon) Relax this assumption.
            expected_names = ["ai", "human"]
            assert all([n == expected_names[i] for i, n in enumerate(config.egocentric_agent_names)])
            self.ai_action_index = 0
            self.human_action_index = 1
        elif self._num_egocentric_agents == 1:
            if config.egocentric_agent_names[0] == "human":
                self.ai_action_index = None
                self.human_action_index = 0
            elif config.egocentric_agent_names[0] == "ai":
                self.ai_action_index = 0
                self.human_action_index = None
            else:
                raise ValueError(f"Unsupported agent name: {config.egocentric_agent_names[0]}")
        else:
            raise ValueError(f"Unsupported number of agents: {self.num_agents}")

    @property
    def base_env(self):
        return self._env.unwrapped

    @property
    def action_spaces(self):
        """
        Returns the action spaces, as outputs from the Dreamer MARL policies.

        Returns:
            list: The action spaces output from the AI and human agents, ordered as [AI, human].
        """
        # AI and human spaces are both steering / throttle / brake
        low = np.array([-1.0] * self.num_allocated_actions)
        high = np.array([1.0] * self.num_allocated_actions)
        ai_space = spaces.Box(low, high)
        ai_space.continuous = True
        human_space = spaces.Box(low, high)
        human_space.continuous = True
        # Action partitioning per agent.  AI first.
        if self._num_egocentric_agents == 2:
            return [ai_space, human_space]
        elif self._num_egocentric_agents == 1:
            if self.ai_action_index is not None:
                return [ai_space]
            else:
                return [human_space]
        else:
            raise ValueError(f"Unsupported number of agents: {self._num_egocentric_agents}")

    @property
    def observation_space(self):
        space = self.base_env.observation_space
        return spaces.Dict(space)

    @property
    def observation_spaces(self):
        space = self.base_env.observation_space
        # Observation partitioning per agent.
        obs_spaces = [space, space] if self._num_egocentric_agents == 2 else [space]
        act_spaces = self.action_spaces
        # Add actions of other agent to respective agent obs space.
        # Note: this is invariant to num_egocentric_agents == 1 or 2, since the agent
        # should be interoperable with either num_egocentric_agent option.
        acts_low = np.concatenate([self.base_env.action_space.low, self.base_env.action_space.low])
        acts_high = np.concatenate([self.base_env.action_space.high, self.base_env.action_space.high])
        for i, (a, o) in enumerate(zip(act_spaces, obs_spaces)):
            low = np.concatenate([space["state"].low, a.low])  # For agent 0
            high = np.concatenate([space["state"].high, a.high])  # For agent 0
            obs_spaces[i] = spaces.Dict(
                image=space["image"],
                state=spaces.Box(low=low, high=high),
            )
        return obs_spaces

    def init_actions(self):
        return [np.zeros(self.num_allocated_actions), np.zeros(self.num_allocated_actions)]

    def allocate_actions(self, policy_actions):
        steer = policy_actions[STEER_INDEX]
        accel = policy_actions[ACCEL_INDEX]
        actions = np.array([steer, max(0.0, accel), -min(0.0, accel)])
        assert len(actions) == len(self.base_env.action_space.low)
        return actions

    def collapse_actions(self, policy_actions):
        """Collapse 3D allocated actions back to 2D policy actions."""
        steer = policy_actions[STEER_INDEX]
        accel = policy_actions[ACCEL_INDEX]
        return np.array([steer, accel])

    def set_inferred_intents(self, intents):
        self.inferred_intents = intents

    def set_label(self, label):
        """Set the current behavior label."""
        self.label = label

    def get_label_dict(self):
        """Return behavior label dictionary."""
        return OrderedDict([(0, "cautious"), (1, "pass")])

    def state_to_observation(self, states, actions):
        """Convert states to observations with action augmentation."""
        obs = self.base_env.update_state_features(states)[0]

        if self._num_egocentric_agents > 1:
            obs["state"] = [
                np.concatenate([obs["state"], np.concatenate([a for j, a in enumerate(actions) if i != j])])
                for i in range(self._num_egocentric_agents)
            ]
            if self.frozen_agent_id is not None:
                unfrozen_agent_id = 1 if self.frozen_agent_id == 0 else 0
                obs["state"][unfrozen_agent_id][-self.num_actions:] = 0.0
        else:
            obs["state"] = [np.concatenate([obs["state"], np.zeros_like(actions[0])])]

        return obs

    def to_int(self, action):
        if len(action.shape) >= 1 and not self.base_env.continuous:
            action = np.argmax(action)
        return action

    def compose_actions(self, actions):
        if self._num_egocentric_agents == 1:
            if self.base_env.continuous:
                return actions[0]
            else:
                return self.to_int(actions[0])
        elif self._num_egocentric_agents == 2:
            if self.base_env.continuous:
                return actions[self.ai_action_index] + actions[self.human_action_index]
            else:
                return (
                    actions[self.ai_action_index] + actions[self.human_action_index] * self.action_spaces[1].n
                )  # Stacked bitvector, since no MultiDiscrete support.
        else:
            raise ValueError(f"Unsupported number of agents: {self.num_agents}")

    def _reset_lap_progress(self):
        """
        Reset the ego agent's checkpoint progress
        """
        self._eval_metric_helpers["last_checkpoint"] = ""
        self._eval_metric_helpers["checkpoint_progress"] = 0

    def _record_and_reset_eval_tracking(self):
        """Record averages/maxes for metrics and reset all intermediate metrics for tracking ego agent's lap time/speed
        This does not overwrite metrics for completed laps, but it deletes metrics for in-progress laps
        """

        # Record average speed metrics based on speed logged at every timestep
        lap_speeds = self._eval_metric_helpers["current_speed"]
        if len(lap_speeds) > 0:
            self.eval_metrics["average_speed"].append(np.mean(lap_speeds))
            self.eval_metrics["max_speed"].append(np.max(lap_speeds))
        # else:  # Debugging print to warn that .reset() was called with no steps.
        #     print(f'Warning -- No speeds recorded for this lap.')

        # Record intervention metrics based on interventions from every timestep
        intervention_norms = self._eval_metric_helpers["current_intervention_norm"]
        if len(intervention_norms) > 0:
            self.eval_metrics["average_intervention_norm"].append(np.mean(intervention_norms))
            self.eval_metrics["max_intervention_norm"].append(np.max(intervention_norms))
        # else:  # Debugging print to warn that .reset() was called with no steps.
        #     print(f'Warning -- No AI actions recorded for this lap.')

        self.eval_metrics["action_discontinuity"].append(self._eval_metric_helpers["action_discontinuity"])

        self._eval_metric_helpers["current_speed"] = []
        self._eval_metric_helpers["current_intervention_norm"] = []
        self._eval_metric_helpers["lap_start_step"] = self._step
        self._eval_metric_helpers["action_discontinuity"] = []

    def log_metrics(self, ai_action):
        # Get velocity based on environment type
        if hasattr(self.base_env, 'agents'):
            # EPIC environment - uses agents with raw_dict
            agent = self.base_env.agents[0]
            agg_v = abs(agent.raw_dict["vx"]) + abs(agent.raw_dict["vy"])
        elif hasattr(self.base_env, 'ego_car'):
            # Box2D environment - uses Car objects with hull.linearVelocity
            ego_velocity = self.base_env.ego_car.hull.linearVelocity
            agg_v = abs(ego_velocity[0]) + abs(ego_velocity[1])
        else:
            # Fallback - skip metrics if environment type unknown
            agg_v = 0.0

        self._eval_metric_helpers["current_speed"].append(agg_v)
        self._eval_metric_helpers["current_intervention_norm"].append(np.linalg.norm(ai_action))

        # agent_x = agent.raw_dict["x"]
        # agent_y = agent.raw_dict["y"]
        seen_all = True
        full_lap = False
        # for k, v in self._checkpoints.items():
        #     if np.sqrt((v[0] - agent_x) ** 2 + (v[1] - agent_y) ** 2) < _DIST_THRESHOLD:
        #         # If this is a new checkpoint, consider it visited. If it's the last checkpoint, ignore this visit.
        #         if self._eval_metric_helpers["last_checkpoint"] != k:
        #             self._eval_metric_helpers["checkpoint_progress"][k] += 1
        #             self._eval_metric_helpers["last_checkpoint"] = k

        #     if self._eval_metric_helpers["checkpoint_progress"][k] < 1:
        #         # If there are any un-visited checkpoints, we can't have done a lap
        #         seen_all = False
        #     elif self._eval_metric_helpers["checkpoint_progress"][k] > 1:
        #         # If a checkpoint counter is > 1, that means we visited at least 1 other checkpoint reaching it
        #         full_lap = True

        if seen_all and full_lap:
            # If one checkpoint is >1 and all checkpoints have been visited, its very likely we did a full lap.
            # WARNING -- This means laps can ONLY start/end on known checkpoints.
            # Record steps_per_lap as curr_step - start_step
            s_step = self._eval_metric_helpers["lap_start_step"]
            self.eval_metrics["steps_per_lap"].append(self._step - s_step)
            self._reset_lap_progress()
            # If we want to record metrics _per lap_, use the below:
            # self._record_and_reset_eval_tracking(agent_id)

    def reset_ado_rails(self):
        """
        Randomly select a new csv from the directory of candidates for the ado rails.
        **WARNING** This reads the file anew, so any update to the file will be reflected here
            Even after launching the training script!
        """
        if self._ref_ado_csv_base is None:  # If there's no directory of rails, there's nothing to load
            return
        new_rails = random.choice(self._ref_ado_csv_fns)
        self._ado_trigger_groups = [self._ado_names]  # Single group by default
        self._ref_ado_csv = pd.read_csv(os.path.join(self._ref_ado_csv_base, new_rails)).to_dict('list')

        # Initialize timestep for each ADO
        for ado_name in self._ado_names:
            timestamp_key = ado_name + " timestamp"
            if timestamp_key in self._ref_ado_csv:
                self._base_ado_timestep[ado_name] = self._ref_ado_csv[timestamp_key][0]
                self._base_ado_timestep[ado_name] += self._ado_headstart * random.random()
            else:
                # Fallback to old single-ADO format
                if "timestamp" in self._ref_ado_csv:
                    self._base_ado_timestep[ado_name] = self._ref_ado_csv["timestamp"][0]
                    self._base_ado_timestep[ado_name] += self._ado_headstart * random.random()
                else:
                    print(f"Warning: No timestamp found for ADO {ado_name}")
                    self._base_ado_timestep[ado_name] = 0.0

    def update_ado_headstart(self):
        if self._headstart_curriculum:
            self._ado_headstart += self._ado_update_rate
            self._ado_headstart = min(self._max_ado_headstart, self._ado_headstart)

    def update_ado_speed_factor(self):
        if self._ado_speed_curriculum:
            self.ado_slowdown_factor -= self._ado_update_rate
            self.ado_slowdown_factor = max(0.0, self.ado_slowdown_factor)
            self._speed_factor = self.make_random_speed_factor()
            print(f"   Ado speed factor {self._speed_factor}")

    def make_random_speed_factor(self):
        """
        Randomly revisit speed factors seen earlier in the training curriculum.
        Otherwise, return the current speed factor based on the current slowdown factor.
        """
        if random.random() < self._speed_factor_randomization_prob:
            return random.uniform(1.0 - self._max_ado_slowdown_factor, 1.0 - self.ado_slowdown_factor)
        return 1.0 - self.ado_slowdown_factor

    def get_rel_pos_ego_frame(self, ado_position, ego_position, ego_car):
        """
        Transform ADO position to ego vehicle's reference frame.

        Args:
            ado_position: ADO position in world frame [x, y]
            ego_position: Ego position in world frame [x, y]
            ego_car: Ego car object with hull.angle attribute

        Returns:
            rel_pos_ego_frame: Relative position [forward, lateral]
                              lateral > 0 means ADO is to the left
                              lateral < 0 means ADO is to the right
        """
        rel_pos = ado_position - ego_position
        rot = R.from_euler("z", ego_car.hull.angle + np.pi / 2).as_matrix()[0:2, 0:2]
        inv_rot = rot.T
        rel_pos_ego_frame = inv_rot.dot(rel_pos)
        return rel_pos_ego_frame

    def shape_auxiliary_driving_rewards(self, reward, acts, over, reward_scalar=1.0):
        """
        Shape the rewards to account for additional factors: bounds and action smoothness to elicit better performance,
        and/or more human-like driving behaviors.

        Args:
            reward (float): The reward from the environment.
            acts (np.array): The actions from the environment.  Dimensions: [num_egocentric_agents, num_actions].
            over (bool): Whether the episode is over.
            reward_scalar (float): Scalar to multiply the final reward by.

        Returns:
            tuple: (list, bool) - The shaped rewards for the ego vehicle and modified over flag.
        """
        action_smoothness_penalty = 0.0
        bound_penalty = 0.0
        collision_penalty = 0.0
        left_right_bias_penalty = 0.0

        acts = np.array(acts)
        # ego_acts = acts[EGO_INDEX]

        ego_car = self.base_env.ego_car
        ego_position = np.array([ego_car.hull.position.x, -ego_car.hull.position.y])
        track = self.base_env.track
        track_left = self.base_env.track_left
        track_right = self.base_env.track_right
        track_xy = np.array(track)[:, 2:4]

        # === Compute track-relative velocity for speed penalties ===
        track_rel_vel = None
        if self._use_speed_penalty or self.config.use_speed_bonus:

            distances = np.linalg.norm(track_xy - ego_position, axis=1)
            curr_tile_idx = np.argmin(distances)
            track_loc = track[curr_tile_idx]
            track_loc_next = track[(curr_tile_idx + 1) % len(track)]
            _, pos_x, pos_y = track_loc[1:4]
            track_heading_angle = np.arctan2(track_loc_next[3] - pos_y, track_loc_next[2] - pos_x)  # Angle of the track at this point.
            track_heading_angle = wrap_angle(track_heading_angle)
            track_rot = R.from_euler("z", track_heading_angle).as_matrix()[0:2, 0:2]
            inv_track_rot = track_rot.T
            velocity = np.array([ego_car.hull.linearVelocity.x, ego_car.hull.linearVelocity.y])
            track_rel_vel = inv_track_rot.dot(velocity)  # Convert to track-relative velocities.


        # === Penalties for collisions ===
        if self._use_collision_penalty:
            ego_position = np.array([ego_car.hull.position.x, -ego_car.hull.position.y])

            # Get all ADO cars
            ado_cars = []
            if self.base_env.num_agents > 1:
                for i in range(self.base_env.num_agents - 1):
                    ado_cars.append(self.base_env.ado_car(i))

            # Check collision with each ADO
            for ado_car in ado_cars:
                ado_position = np.array([ado_car.hull.position.x, -ado_car.hull.position.y])
                dist = np.linalg.norm(ado_position - ego_position)

                # Get thresholds from config with defaults
                inner_threshold = getattr(self.config, 'inner_coll_dist_threshold', 4.0)
                outer_threshold = getattr(self.config, 'outer_coll_dist_threshold', 70.0)
                inner_penalty = getattr(self.config, 'inner_coll_penalty', 10000.0)
                outer_penalty = getattr(self.config, 'outer_coll_penalty', 100.0)

                if dist < inner_threshold:
                    # Inner collision - terminate episode
                    collision_penalty = inner_penalty
                    over = True
                    if self.base_env.verbose:
                        print(f"Collision! Distance: {dist:.2f}, terminating episode")
                    break  # No need to check other ADOs
                elif dist < outer_threshold:
                    # Near collision - penalty but continue
                    collision_penalty = max(collision_penalty, outer_penalty)  # Use max if multiple ADOs nearby
                    if self.base_env.verbose:
                        print(f"Near-collision penalty: {collision_penalty}, distance: {dist:.2f}")

        # === Penalties for left/right bias ===
        # Penalize if ego is biased towards the wrong side of the nearest ADO car.
        use_left_bias_penalty = getattr(self.config, 'use_left_bias_penalty', False)
        use_right_bias_penalty = getattr(self.config, 'use_right_bias_penalty', False)

        if (use_left_bias_penalty or use_right_bias_penalty) and self.base_env.num_agents > 1:
            ego_position = np.array([ego_car.hull.position.x, -ego_car.hull.position.y])
            left_right_bias_dist_threshold = getattr(self.config, 'left_right_bias_dist_threshold', 8.0)
            left_right_bias_penalty_value = getattr(self.config, 'left_right_bias_penalty', 1000.0)
            left_right_penality_moving_ados_only = getattr(self.config, 'left_right_penality_moving_ados_only', True)

            # Get all ADO cars
            ado_cars = []
            for i in range(self.base_env.num_agents - 1):
                ado_cars.append(self.base_env.ado_car(i))

            # Check left/right bias with each ADO
            for ado_car in ado_cars:
                # Skip stationary ADOs if configured
                if left_right_penality_moving_ados_only and np.linalg.norm(ado_car.hull.linearVelocity) < 0.1:
                    continue

                ado_position = np.array([ado_car.hull.position.x, -ado_car.hull.position.y])
                dist = np.linalg.norm(ado_position - ego_position)

                if dist < left_right_bias_dist_threshold:
                    rel_pos_ego_frame = self.get_rel_pos_ego_frame(ado_position, ego_position, ego_car)

                    if rel_pos_ego_frame[1] > 0:  # ADO is to the left of ego car
                        if use_left_bias_penalty:
                            left_right_bias_penalty += left_right_bias_penalty_value
                            if self.base_env.verbose:
                                print(f"Left bias penalty applied: ADO to the left at distance {dist:.2f}")

                    elif rel_pos_ego_frame[1] < 0:  # ADO is to the right of ego car
                        if use_right_bias_penalty:
                            left_right_bias_penalty += left_right_bias_penalty_value
                            if self.base_env.verbose:
                                print(f"Right bias penalty applied: ADO to the right at distance {dist:.2f}")

                    break  # Only check the nearest ADO

        # === Penalties governing human or AI going out of bounds ===
        bound_penalty = 0.
        if self.config.use_out_of_bounds_penalty:
            # Heavily penalize if ego goes out of bounds
            ego_distances = np.linalg.norm(np.array(track)[:, 2:4] - np.array(ego_car.hull.position), axis=1)
            ego_track_index = np.argmin(ego_distances)
            # TODO(jon): Add documentation explaining why the minus pi is needed here.
            left_bound = self.base_env._transform_points(
                (ego_car.hull.angle - np.pi, *ego_car.hull.position), np.array([track_left[ego_track_index]])
            )
            right_bound = self.base_env._transform_points(
                (ego_car.hull.angle - np.pi, *ego_car.hull.position), np.array([track_right[ego_track_index]])
            )
            if (np.sign(left_bound[0][0]) == np.sign(right_bound[0][0])) and (
                np.sign(left_bound[0][1]) == np.sign(right_bound[0][1])
            ):
                bound_penalty = self.config.out_of_bounds_penalty
                # Additional metrics for wandb logging
                self.out_of_bound_count += 1

        # === Penalties governing human or AI actuation smoothness ===
        # N.B. These were especially necessary in preventing the AI agent's steering interventions from being too noisy and jerky.
        steering_difference = 0.0
        if self._last_acts_for_reward_shaping is not None and self._use_action_diff_penalty:
            # Steering action penalty.
            if self._num_egocentric_agents == 2:  # Penalize unsmooth AI actions.
                steering_difference = (
                    acts.take(self.ai_action_index)[STEER_INDEX]
                    - self._last_acts_for_reward_shaping[self.ai_action_index][STEER_INDEX]
                )
            elif self._num_egocentric_agents == 1:  # Penalize unsmooth human actions.
                steering_difference = (
                    acts.take(self.human_action_index)[STEER_INDEX]
                    - self._last_acts_for_reward_shaping[self.human_action_index][STEER_INDEX]
                )
            else:
                raise ValueError(f"Unsupported number of agents: {self._num_egocentric_agents}")
            if self._global_step is not None:
                multiplier = min(self._max_steering_penalty, self._steering_penalty_update_rate * self._global_step)
                if self._global_step % 100 == 0:
                    print(f"  steering penalty multiplier {multiplier}")
                action_smoothness_penalty = multiplier * np.linalg.norm(steering_difference)
        # self._eval_metric_helpers["action_discontinuity"].append(np.square(steering_difference / 2.0))
        self._last_acts_for_reward_shaping = acts

        # === Compute distance flags for speed penalties and bonuses ===
        within_inner_coll_dist = False
        within_outer_coll_dist = False
        within_near_ado_dist = False

        if self.base_env.num_agents > 1:
            ego_position = np.array([ego_car.hull.position.x, -ego_car.hull.position.y])
            inner_threshold = getattr(self.config, 'inner_coll_dist_threshold', 4.0)
            outer_threshold = getattr(self.config, 'outer_coll_dist_threshold', 70.0)
            near_ado_threshold = getattr(self.config, 'near_ado_speed_penalty_distance', 15.0)

            for i in range(self.base_env.num_agents - 1):
                ado_car = self.base_env.ado_car(i)
                ado_position = np.array([ado_car.hull.position.x, -ado_car.hull.position.y])
                dist = np.linalg.norm(ado_position - ego_position)

                if dist < inner_threshold:
                    within_inner_coll_dist = True
                if dist < outer_threshold:
                    within_outer_coll_dist = True
                if dist < near_ado_threshold:
                    within_near_ado_dist = True

        # === Penalties for extremely low speeds ===
        speed_penalty = 0.
        if track_rel_vel is not None and track_rel_vel[0] < 0.5 and not within_inner_coll_dist:
            speed_penalty = self.config.default_low_speed_penalty
            if self._use_speed_penalty:
                speed_penalty = self.config.additional_low_speed_penalty

        # === Penalty for high speeds near ado ===
        if track_rel_vel is not None and self.config.use_near_ado_speed_penalty and within_near_ado_dist:
            speed_penalty += self.config.near_ado_speed_penalty_multiplier * track_rel_vel[0]

        # === Bonus for higher speeds ===
        speed_bonus = 0.
        if track_rel_vel is not None and self.config.use_speed_bonus:
            if self._use_collision_penalty:
                if not within_outer_coll_dist:
                    speed_bonus = self.config.speed_bonus_multiplier * track_rel_vel[0]
            elif not within_inner_coll_dist:
                speed_bonus = self.config.speed_bonus_multiplier * track_rel_vel[0]

        # === Combine all the rewards and penalties ===
        bonuses_and_penalties = speed_bonus - (action_smoothness_penalty + bound_penalty + collision_penalty + speed_penalty + left_right_bias_penalty)
        if self.base_env.verbose:
            print(f"reward: {reward}, bonuses_and_penalties: {bonuses_and_penalties} (speed_bonus: {speed_bonus}, action_smoothness_penalty: {action_smoothness_penalty}, bound_penalty: {bound_penalty}, collision_penalty: {collision_penalty}, speed_penalty: {speed_penalty}), left_right_bias_penalty: {left_right_bias_penalty})")
        self._eval_metric_helpers["reward_components"] = {
            "base_reward": reward,
            "speed_bonus": speed_bonus * reward_scalar,
            "action_smoothness_penalty": -action_smoothness_penalty * reward_scalar,
            "bound_penalty": -bound_penalty * reward_scalar,
            "collision_penalty": -collision_penalty * reward_scalar,
            "speed_penalty": -speed_penalty * reward_scalar,
            "left_right_bias_penalty": -left_right_bias_penalty * reward_scalar,
        }

        if self._num_egocentric_agents == 2:
            human_acts_reward = np.linalg.norm(
                acts[self.human_action_index] - acts[self.ai_action_index]
            )  # Reward for adhering to human actions.
            result = [reward, reward]
            result[self.ai_action_index] += (bonuses_and_penalties + human_acts_reward) * reward_scalar
            self._eval_metric_helpers["reward_components"]["human_acts_bonus"] = human_acts_reward
        elif self._num_egocentric_agents == 1:
            result = [reward + bonuses_and_penalties * reward_scalar]
        else:
            raise ValueError(f"Unsupported number of agents: {self._num_egocentric_agents}")

        # Additional metrics for wandb logging
        if wandb.run and not wandb.run.disabled:  # Ensure wandb is enabled
            wandb.log({"out-of-bounds occurrences": self.out_of_bound_count})
            wandb.log({"incorrect-left-steer count": self.incorrect_left_steer_count})
            wandb.log({"incorrect-right-steer count": self.incorrect_right_steer_count})
            wandb.log({"incorrect-steering-angle count": self.incorrect_steering_angle_count})
            for ado_id, collisions_count in self.collisions_per_ado_id.items():
                wandb.log({f"ado{ado_id} collisions count": collisions_count})

        # Update reward_component_names to match the actual components in this step
        # This ensures consistency even if the configuration changes
        self.reward_component_names = list(self._eval_metric_helpers["reward_components"].keys())

        return result, over

    def step(self, actions):
        """
        Steps the environment forward.  Step is responsible for blending the actions of the human and AI agents and
        allocating those to the ego vehicle, then stepping both the ego and ado vehicles.

        Args:
            actions (list): A list of actions for the AI and human agents.

        Returns:
            dict: The observation from the environment.
            list: A list of rewards for the AI and human.
            bool: Whether the episode is done.
            dict: The info dictionary.
        """
        action = self.compose_actions(actions)

        policy_action = action.astype(np.float64)  # Here, agent 0 refers to car 0 (ego car).
        vehicle_action = self.allocate_actions(policy_action)

        total_reward = [0.0] * self.num_agents

        for repeat in range(self._repeat):
            # print(vehicle_action)
            obs_dict, reward_item, over, info_item = self.base_env.step(vehicle_action)
            # print(f" obs_dict state len {len(obs_dict['state'])}, state shapes {[o.shape for o in obs_dict['state']]}")
            obs = obs_dict

            # Boost the base driving reward to more strongly incentivize driving.
            reward = reward_item * 100.0

            info = info_item
            # ado_state = self.move_ado()
            # Parse for agent 0 only.
            # print(f"obs_dict: {obs_dict}")
            # If ego data looks like it exists twice (near zero diff)
            # print(f"overwriting ego with ado ? {self.overwrite_ego_with_ado}")
            # if sum(abs(obs[-6:-3] - obs[2:5])) < 1e-6 and self.overwrite_ego_with_ado:
            #     # Overwrite duplicate multi-agent ego data with ado dynamics:
            #     obs = self.inject_ado_state_data(obs, ado_state)
            #     if not self._has_ado_overritten_ego:
            #         print("WARNING: Ego state has been overwritten by Ado dynamics in this step, " + \
            #               "but not the previous one.  Results may be erroneous!!")
            #         # TODO(jon) Decide if we want an assert here instead.
            #     self._has_ado_overritten_ego = True
            # else:
            #     self._has_ado_overritten_ego = False

            # The first time the agent's progress goes above 50, start counting global steps.  This will,
            # in turn, start the action penalty scheduler, which is meant to prevent wide swings in steering angle.
            # self._running_progress += info[0]["track_info"]["progress"]
            # if self._running_progress > self.config.progress_threshold_for_steering_penalty and self._global_step is None:
            #     self._global_step = 0

            # We need to make an explicit obs to allow shape_rewards to evaluate optimal actions and rewards from the optimal inferred human model.
            # The obs from the one we use below because it does not contain rewards.  That, of course, will be populated by shape_rewards;
            # however, we don't need these for evaluating the human model, permitting us to zero these out for purposes of human model evaluation.
            obs_struct_no_reward = [deepcopy(obs)]
            obs_struct_no_reward[0].update(
                {
                    "is_terminal": np.array(False) if over is None else np.array(over),
                    "is_first": np.array(False),
                    "reward": [0.0, 0.0],
                }
            )
            # Append the action of the other agents
            if self._num_egocentric_agents > 1:
                obs_struct_no_reward[0]["state"] = [
                    np.concatenate(
                        [obs_struct_no_reward[0]["state"], np.concatenate([a for j, a in enumerate(actions) if i != j])]
                    )
                    for i in range(self._num_egocentric_agents)
                ]
                if self.frozen_agent_id is not None:
                    # Zero out the actions of the frozen agent, as observed by the unfrozen agent.
                    unfrozen_agent_id = 1 if self.frozen_agent_id == 0 else 0

                    obs_struct_no_reward[0]["state"][unfrozen_agent_id][-self.num_actions :] = 0.0
            else:  # Only one agent.
                # Note: We assume capacity to observe one other agent; however, we don't have access to that agent, so we just pad the actions with zeros.
                # print(obs_struct_no_reward[0])
                # print(actions[0])
                obs_struct_no_reward[0]["state"] = [
                    np.concatenate([obs_struct_no_reward[0]["state"], np.zeros_like(actions[0])])
                ]  # Append a null action to enable eventual use in a two-egocentric agent env.

            obs_no_reward = {}
            for k in obs_struct_no_reward[0]:
                try:
                    obs_no_reward[k] = (
                        [o[k] for o in obs_struct_no_reward]
                        if isinstance(obs_struct_no_reward[0][k], tuple)
                        else np.stack([o[k] for o in obs_struct_no_reward])
                    )
                except ValueError as e:
                    print(e)
                    print(" Can't append a mismatched array size! Something's up with envs")
                    exit()

            reward, over = self.shape_auxiliary_driving_rewards(reward, actions, over, reward_scalar=100.0)

            self._step += 1
            if self._global_step is not None:
                self._global_step += 1

            if not isinstance(reward, float):
                for i, r in enumerate(reward):
                    total_reward[i] += r or 0
            else:
                for i in range(self.num_agents):
                    total_reward[i] += reward or 0

            if over:
                break

        # obs = {
        #     "image": np.array([[[0, 0, 0]]], dtype=np.uint8),
        #     "state": obs,
        # }
        obs["image"] = np.array([[[0, 0, 0]]], dtype=np.uint8)

        # Replicate the observation for all agents and append the action of the other agents
        # print(f" prior obs state len {len(obs['state'])}, state shapes {[o.shape for o in obs['state']]}")
        if self._num_egocentric_agents > 1:
            obs["state"] = [
                np.concatenate([obs["state"], np.concatenate([a for j, a in enumerate(actions) if i != j])])
                for i in range(self._num_egocentric_agents)
            ]
            if self.frozen_agent_id is not None:
                # Zero out the actions of the frozen agent, as observed by the unfrozen agent.
                unfrozen_agent_id = 1 if self.frozen_agent_id == 0 else 0

                obs["state"][unfrozen_agent_id][-self.num_actions :] = 0.0
        else:  # Only one agent.
            # print(f" obs state len {len(obs['state'])}, state shapes {[o.shape for o in obs['state']]}")
            # print(f" action shape {actions[0].shape}")
            # Note: We assume capacity to observe one other agent; however, we don't have access to that agent, so we just pad the actions with zeros.
            obs["state"] = [
                np.concatenate([obs["state"], np.zeros_like(actions[0])])
            ]  # Append a null action to enable eventual use in a two-egocentric agent env.

        self._done = over or (self._length and self._step >= self._length)

        if self.ai_action_index is not None:
            self.log_metrics(actions[self.ai_action_index])
        info.update(
            {
                # "agent_actions": [[0.0, 0.0]] * self.num_agents,
                # "agent_states": [(0.0, 0.0, 0.0)] * self.num_agents,
                "speed": self._eval_metric_helpers["current_speed"][-1] if self.ai_action_index is not None else 0,
                "intervention_norm": (
                    self._eval_metric_helpers["current_intervention_norm"][-1]
                    if self.ai_action_index is not None
                    else 0
                ),
                "progress": self._running_progress,
                "collisions": 0.0,
                "intent_specific_reward": 0.0,
                "reward_components": list(self._eval_metric_helpers["reward_components"].values()),
                # "collision": int(self.base_env.agents[0].has_collided()),  # Did ego collide?
                # "action_discontinuity": self._eval_metric_helpers["action_discontinuity"][-1],
            }
        )

        obs["is_terminal"] = False if over is None else over
        obs["is_first"] = False
        done = False if over is None else over

        if self.base_env.verbose:
            print(f" wrapper reward: {total_reward}")
        return obs, total_reward, done, info

    def reset(self, **kwargs):
        self._done = False
        self._step = 0

        result = self.base_env.reset()
        default_state = result[0]["state"]  # For agent 0
        default_image = np.array([[[0, 0, 0]]], dtype=np.uint8)
        obs = result[0]

        obs = {
            "image": default_image,
            "state": default_state,
            "is_terminal": np.array(False),
            "is_first": np.array(True),
        }
        self._record_and_reset_eval_tracking()
        self._reset_lap_progress()

        obs["is_terminal"] = False
        obs["is_first"] = True

        # Assume the first action is zero; use this as the observation.  These should correspond to a resonable null
        # action (e.g. initial condition of the Box2d Car model, default (null) AI input).
        default_action = self.init_actions()
        obs["state"] = [
            np.concatenate([obs["state"], default_action[EGO_INDEX]]) for _ in range(self._num_egocentric_agents)
        ]

        self.road_poly = self.base_env.road_poly

        return obs

    def close(self):
        return self.base_env.close()
