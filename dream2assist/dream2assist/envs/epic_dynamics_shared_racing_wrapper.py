from collections import OrderedDict
from copy import deepcopy
from glob import glob
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
from pandas import merge, read_csv
from pathlib import Path
import random
from scipy.spatial.transform import Rotation as R
import math
import torch
import wandb

import envs.epic_dynamics_shared_racing
import tools
from util import count_steps, find_free_port, make_absolute_path, PathInterpolator, wrap_angle

_DIST_THRESHOLD = 15  # Threshold to hit a checkpoint on the track

STEER_INDEX = 0
ACCEL_INDEX = 1

CAR_LENGTH = 7.0  # Assumed length of the car, in meters.


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

    logdir = Path(config.logdir + "_tmp").expanduser()
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

    env = EpicDynamicsSharedRacingWrapper(
        config=config,
        render_mode=None if mode == "eval" else "no_render",
        track_csv=str(make_absolute_path(config.track_csv)),
        dreamer_policy=eval_policy,
        human_agents=labeled_human_agents,
    )
    renderer = functools.partial(Render, env)
    return env, renderer


def make_human_agents(config, task, env_config_overrides, model_cls, port=None):
    # TODO: Update me.
    config.egocentric_agent_names = ["human"]
    config.human_agent_paths = []
    config.frozen_agent_id = 0  # Ensure the weights of this agent are not updated.
    env_config_overrides_human_only = deepcopy(env_config_overrides)

    env_config_overrides_human_only["sim_config"]["port"] = port
    if port is None:
        env_config_overrides_human_only["sim_config"]["port"] = find_free_port()
    # TODO(jon): If init_carla is true, then we're specifying a port in `make_env_config_overrides` and another one here.
    # We should clean this up and consolidate ports.
    env_config_overrides_human_only["sim_config"]["init_carla"] = True

    human_only_env = EpicDynamicsSharedRacingWrapper(
        task,
        config.action_repeat,
        config.size,
        config=config,
        track_csv=str(make_absolute_path(config.track_csv)),
        config_name=config.multicar_racing_config_name,
        config_overrides=env_config_overrides_human_only,
    )

    logdir = Path(config.logdir + "_tmp").expanduser()
    logdir.mkdir(parents=True, exist_ok=True)
    step = count_steps(config.traindir)
    tmp_logger = tools.Logger(logdir, config.action_repeat * step)
    obs_spaces = human_only_env.observation_spaces
    act_spaces = human_only_env.action_spaces
    labeled_human_agents = {}
    for path in config.human_ego_agent_paths:
        human_agents = model_cls(obs_spaces, act_spaces, config, tmp_logger).to(config.device)
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


class EpicDynamicsSharedRacingWrapper:
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

        self._use_action_diff_penalty = config.use_action_diff_penalty
        self._use_collision_penalty = config.use_collision_penalty
        self._use_speed_penalty = config.use_speed_penalty
        self._max_steering_penalty = config.max_steering_penalty
        self._steering_penalty_update_rate = config.steering_penalty_update_rate
        self._speed_factor_randomization_prob = config.speed_factor_randomization_prob
        self.track_df = str(make_absolute_path(config.track_csv))

        # === Initialization ===
        self._done = True
        self._step = 0
        self._global_step = None
        self._num_egocentric_agents = len(config.egocentric_agent_names)
        self._ref_ado_csv = None
        self._base_ado_timestep = {}
        self._ref_ado_csv_base = None
        self._ado_speed_factor_lb = config.ado_speed_factor_lb
        self._ado_speed_factor_ub = config.ado_speed_factor_ub
        self._ado_speed_factor = 1.0
        self._last_acts_for_reward_shaping = None
        self._running_progress = 0.0

        # === Inference-related parameters ===
        behavior_labels = config.behavior_labels if hasattr(config, "behavior_labels") else []
        self.label_dict = {i: v for i, v in enumerate(behavior_labels)}
        self.label = None  # Must be an integer index into behavior_labels.
        # Currently, we assume that we'll zero out the AI action from the perspective of the human ego driver.
        # This will need to change if we decide to interleave with a reactive human.
        self.frozen_agent_id = config.frozen_agent_id if config else None
        self.human_agents = human_agents
        self._human_agent_state = (
            None  # Note: be careful to reset this if an agent's model is switched due to inference.
        )
        self._last_inferred_intent = 0
        self.inferred_intents = None
        self.use_intent = config.use_intent if config else False

        self.num_ados = 0
        if hasattr(config, "ado_rails"):
            # Note that these two remain constant throughout training, even if different ado rails populate upon reset.
            self._ado_names = config.ado_names
            self.num_ados = len(self._ado_names)

            num_agents = self.num_ados + 1  # +1 for the ego agent.
            self._ref_ado_csv_base = str(make_absolute_path(config.ado_rails))
            self._ref_ado_csv_fns = []
            self._ado_trigger_groups = None  # This is a list of ado names that will be used to trigger the ado rails.
            for fn in os.listdir(self._ref_ado_csv_base):
                if fn.endswith(".csv"):
                    self._ref_ado_csv_fns.append(fn)
            self._ado_headstart = config.max_ado_headstart
            if config.use_demo_ado_rails:
                self.reset_demo_ado_rails()  # Populates self._ref_ado_csv and self._base_ado_timestep.
            else:
                self.reset_ado_rails()  # Populates self._ref_ado_csv and self._base_ado_timestep.

        with self.LOCK:
            self._env = gym.make(
                "EpicDynamicsSharedRacing-v0",
                config=config,
                render_mode=render_mode,
                num_agents=num_agents,
                continuous=True,
                track_csv=str(make_absolute_path(config.track_csv)),
                dreamer_policy=human_agents,
                start_position=start_position,
                ado_rails=self._ref_ado_csv if self._ref_ado_csv is not None else None,
                ado_names=self._ado_names if self._ref_ado_csv is not None else None,
                ado_trigger_groups=self._ado_trigger_groups if self._ref_ado_csv is not None else None,
                ado_speed_factors=[self.make_random_speed_factor() for _ in range(len(self._ado_trigger_groups))] if self._ref_ado_csv is not None else None,
                verbose=config.is_verbose,
                pct_track_to_visit=config.pct_track_to_visit,
            )

        self._is_verbose = config.is_verbose if config else False
        self.num_agents = self.base_env.num_agents
        self.action_space = self.base_env.action_space
        self.num_actions = 2  # This is number of *un-allocated* actions. e.g. steering and accel.
        self.num_allocated_actions = 3  # This is number of *allocated* actions. e.g. steering, throttle, brake.

        self.actions = self.base_env.actions  # These are the *allocated* actions.
        self.continuous = self.base_env.continuous
        self.road_poly = self.base_env.road_poly

        self.vehicle_data = []  # For BEV plotting.

        self.reward_component_names = None

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
            "collisions": 0,
            "current_speed": [],
            "last_checkpoint": "",
            "current_intervention_norm": [],
            "action_discontinuity": [],
            "intent_specific_reward": 0,
            "reward_components": None,
        }

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
    def ego_index(self):
        return self._env.unwrapped.ego_index

    @property
    def action_spaces(self):
        """
        Returns the action spaces, as outputs from the Dreamer MARL policies.

        Returns:
            list: The action spaces output from the AI and human agents, ordered as [AI, human].
        """
        # AI and human spaces are both steering / throttle
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

    def state_to_observation(self, states, actions):
        obs = self.base_env.update_state_features(states)[0]

        # Prepare ego observation
        obs = {
            "image": obs[0]["image"],
            "state": np.concatenate([v["state"] for v in obs]),
        }

        # Replicate the observation for all agents and append the action of the other agents
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
            # Note: We assume capacity to observe one other egocentric agent; however, we don't have access to that agent, so we just pad the actions with zeros.
            print("state_to_observation: No human agent found!  Zeroing out the human action in the observation.")
            temp = []
            for i in range(self.num_agents):
                temp.append(
                    np.concatenate([obs["state"][i], np.zeros_like(actions[0])])
                )  # Append a null action to keep the policy in distribution.
            obs["state"] = temp

        return obs

    def init_actions(self):
        return [np.zeros(self.num_allocated_actions), np.zeros(self.num_allocated_actions)]

    def allocate_actions(self, policy_actions):
        steer = policy_actions[STEER_INDEX]
        accel = policy_actions[ACCEL_INDEX]
        actions = np.array([steer, max(0.0, accel), -min(0.0, accel)])
        assert len(actions) == len(self.base_env.action_space.low)
        return actions

    def collapse_actions(self, policy_actions):
        steer = policy_actions[STEER_INDEX]
        accel = policy_actions[ACCEL_INDEX]
        actions = np.array([steer, accel])
        return actions

    def set_inferred_intents(self, intents):
        self.inferred_intents = intents

    def set_label(self, label):
        self.label = int(label)

    def get_label_dict(self):
        return self.label_dict

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
            raise ValueError(f"Unsupported number of agents: {self.num_egocentric_agents}")

    def _reset_lap_progress(self):
        """
        Reset the ego agent's checkpoint progress
        """
        self._eval_metric_helpers["last_checkpoint"] = ""
        self._eval_metric_helpers["checkpoint_progress"] = 0
        self._eval_metric_helpers["collisions"] = 0
        self._eval_metric_helpers["intent_specific_reward"] = 0.0

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
        self._eval_metric_helpers["reward_components"] = []

    def log_metrics(self, ai_action):
        ego_speed = np.linalg.norm(self.base_env.ego_velocity)
        self._eval_metric_helpers["current_speed"].append(ego_speed)

        self._eval_metric_helpers["current_intervention_norm"].append(np.linalg.norm(ai_action))

        # TODO(jon) Enable logging lap-level metrics as in
        # https://github.com/ToyotaResearchInstitute/shared-decision-making/blob/5980e56/shared_control_racing/human_ai_racing_carla_wrapper.py#L634

    def reset_demo_ado_rails(self):
        """
        For demo ado rails, we want to reset rails named east/west to random ones independently, and also reset the carousel ados.
        """
        if self._ref_ado_csv_base is None:  # If there's no directory of rails, there's nothing to load
            return
        carousel_csvs_keyword = "carousel"
        east_csvs_keyword = "east"
        west_csvs_keyword = "west"
        all_ref_ado_csv_fns = {
            carousel_csvs_keyword: glob(os.path.join(self._ref_ado_csv_base, f"*{carousel_csvs_keyword}*.csv")),
            east_csvs_keyword: glob(os.path.join(self._ref_ado_csv_base, f"*{east_csvs_keyword}*.csv")),
            west_csvs_keyword: glob(os.path.join(self._ref_ado_csv_base, f"*{west_csvs_keyword}*.csv")),
        }
        self._ado_trigger_groups = []
        # Randomly select a new csv from the directory of candidates for the ado rails.
        self._ref_ado_csv = {}
        for k, fns in all_ref_ado_csv_fns.items():
            print(f"Found {len(fns)} csvs for keyword {k}.")
            if len(fns) == 0:
                continue
            new_rail_file = random.choice(fns)
            new_rails = read_csv(os.path.join(self._ref_ado_csv_base, new_rail_file)).to_dict("list")
            self._ado_trigger_groups.append(
                [v for v in self._ado_names if any([v in k for k in list(new_rails.keys())])]
            )
            self._ref_ado_csv.update(new_rails)
        for name in self._ado_names:
            if name + " timestamp" not in self._ref_ado_csv:
                raise ValueError(
                    f"Ado name {name} not found in csv columns {self._ref_ado_csv.keys()}!  Update the ado names in the config."
                )
            # self._ref_ado_csv[name + " timestamp"] = round(
            #     self._ref_ado_csv[name + " timestamp"], 5
            # )  # Round to 5 places to find nearest neighbors
            self._base_ado_timestep[name] = self._ref_ado_csv[name + " timestamp"][0]
            self._base_ado_timestep[name] += self._ado_headstart * random.random()

    def reset_ado_rails(self):
        """
        Randomly select a new csv from the directory of candidates for the ado rails.
        **WARNING** This reads the file anew, so any update to the file will be reflected here
            Even after launching the training script!
        """
        if self._ref_ado_csv_base is None:  # If there's no directory of rails, there's nothing to load
            return
        new_rails = random.choice(self._ref_ado_csv_fns)
        self._ado_trigger_groups = (
            self._ado_names
        )  # This is a list of ado names that will be used to trigger the ado rails.
        name = self._ado_names[0]
        self._ref_ado_csv = read_csv(os.path.join(self._ref_ado_csv_base, new_rails)).to_dict()
        # self._ref_ado_csv[name + " timestamp"] = round(
        #     self._ref_ado_csv[name + " timestamp"], 5
        # )  # Round to 5 places to find nearest neighbors
        self._base_ado_timestep[name] = self._ref_ado_csv[name + " timestamp"][0]
        self._base_ado_timestep[name] += self._ado_headstart * random.random()  # random time addition

    def make_random_speed_factor(self):
        """
        Randomly revisit speed factors seen earlier in the training curriculum.
        Otherwise, return the current speed factor based on the current slowdown factor.
        """
        if random.random() < self._speed_factor_randomization_prob:
            return random.uniform(self._ado_speed_factor_lb, self._ado_speed_factor_ub)
        return 1.0

    def get_rel_pos_ego_frame(self, ado_position, ego_position, ego_car):
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
            reward_scalar (float): A scalar to multiply the shaped reward by.

        Returns:
            list: The shaped rewards for the ego vehicle.
        """
        acts = np.array(acts)

        ego_car = self.base_env.ego_car
        track = self.base_env.track
        track_left = self.base_env.track_left
        track_right = self.base_env.track_right

        ego_position = np.array([ego_car.hull.position.x, ego_car.hull.position.y])
        rot = R.from_euler("z", ego_car.hull.angle + np.pi / 2).as_matrix()[0:2, 0:2]
        front_offset = rot.dot(np.array([CAR_LENGTH / 3, 0]))
        ego_position_f = np.array([ego_car.hull.position.x + front_offset[0], ego_car.hull.position.y + front_offset[1]])
        ego_position_r = np.array([ego_car.hull.position.x - front_offset[0], ego_car.hull.position.y - front_offset[1]])

        track_rel_vel = None
        if self._use_speed_penalty or self.config.use_speed_bonus:
            track_xy = np.array(track)[:, 2:4]
            distances = np.linalg.norm(track_xy - ego_position, axis=1)
            curr_tile_idx = np.argmin(distances)
            track_loc = track[curr_tile_idx]
            track_loc_next = track[(curr_tile_idx + 1) % len(track)]
            _, pos_x, pos_y = track_loc[1:4]
            track_heading_angle = np.arctan2(track_loc_next[3] - pos_y, track_loc_next[2] - pos_x)  # Angle of the track at this point.
            track_heading_angle = wrap_angle(track_heading_angle)
            track_rot = R.from_euler("z", track_heading_angle).as_matrix()[0:2, 0:2]
            inv_track_rot = track_rot.T
            velocity = self.base_env.ego_velocity
            track_rel_vel = inv_track_rot.dot(velocity)  # Convert to track-relative velocities.

        # === Penalties for collisions ===
        # If the distance between the ego car and ado car is below threshold, then we penalize the reward
        collision_penalty = 0.
        ado_cars = (
            [self.base_env.ado_car(i) for i in range(self.base_env.num_agents - 1)]
            if self.base_env.num_agents > 1
            else None
        )

        within_inner_coll_dist = False
        within_outer_coll_dist = False
        within_near_ado_dist = False
        if ado_cars is not None:
            for ado_car_idx, ado_car in enumerate(ado_cars):
                # Check collisions using a three-circle approximation of the two cars (center, front, rear)
                ado_position = np.array([ado_car.hull.position.x, ado_car.hull.position.y])
                dist1 = min(np.linalg.norm(ado_position - ego_position), np.linalg.norm(ado_position - ego_position_f), np.linalg.norm(ado_position - ego_position_r))
                rot = R.from_euler("z", ado_car.hull.angle + np.pi / 2).as_matrix()[0:2, 0:2]
                front_offset = rot.dot(np.array([CAR_LENGTH / 3, 0]))
                ado_position += ado_position + front_offset
                dist2 = min(np.linalg.norm(ado_position - ego_position), np.linalg.norm(ado_position - ego_position_f), np.linalg.norm(ado_position - ego_position_r))
                ado_position -= - 2 * front_offset
                dist3 = min(np.linalg.norm(ado_position - ego_position), np.linalg.norm(ado_position - ego_position_f), np.linalg.norm(ado_position - ego_position_r))
                dist = min(dist1, dist2, dist3)
                if dist < self.config.near_ado_speed_penalty_distance:
                    within_near_ado_dist = True
                    if self._is_verbose:
                        print(f"Ego-ado within NEAR ado distance: {dist}")
                if dist < self.config.inner_coll_dist_threshold:
                    self._eval_metric_helpers["collisions"] += 1
                    if self.collisions_per_ado_id.get(ado_car_idx) is not None:
                        self.collisions_per_ado_id[ado_car_idx] += 1
                    else:
                        self.collisions_per_ado_id[ado_car_idx] = 1
                    within_inner_coll_dist = True
                    within_outer_coll_dist = True
                    if self._is_verbose:
                        print(f"Ego-ado within INNER collision distance: {dist}")
                elif dist < self.config.outer_coll_dist_threshold:
                    within_outer_coll_dist = True
                    if self._is_verbose:
                        print(f"Ego-ado within OUTER collision distance: {dist}")
                if self._use_collision_penalty:
                    if within_inner_coll_dist:
                        collision_penalty = self.config.inner_coll_penalty
                        over = True if self.config.end_episode_on_collision else over
                        if self._is_verbose:
                            print(f"Collision penalty: {collision_penalty}")
                        break  # No need to check other ados if we're already within inner distance.
                    elif within_outer_coll_dist:
                        collision_penalty = self.config.outer_coll_penalty
                        if self._is_verbose:
                            print(f"Collision penalty: {collision_penalty}")
                        break  # No need to check other ados if we're already within outer distance.
                else:
                    if within_inner_coll_dist:
                        collision_penalty = self.config.inner_coll_penalty
                        over = True if self.config.end_episode_on_collision else over
                        if self._is_verbose:
                            print(f"Collision penalty: {collision_penalty}")
                        break  # No need to check other ados if we're already within inner distance.

        # === Left and right biased penalties ===
        left_right_bias_penalty = 0.
        # Penalize if ego is biased towards the left or right of the nearest ado car.
        if ado_cars is not None:
            for ado_car in ado_cars:
                if self.config.left_right_penality_moving_ados_only and np.linalg.norm(ado_car.hull.linearVelocity) < 0.1:
                    continue  # Skip stationary ados.
                ado_position = np.array([ado_car.hull.position.x, ado_car.hull.position.y])
                dist = np.linalg.norm(ado_position - ego_position)
                if dist < self.config.left_right_bias_dist_threshold:
                    rel_pos_ego_frame = self.get_rel_pos_ego_frame(ado_position, ego_position, ego_car)
                    if rel_pos_ego_frame[1] > 0:
                        label_keys = np.array(list(self.get_label_dict().keys()))
                        if label_keys.size != 0:
                            ai_index = 0
                            inferred_intent = self.inferred_intents[ai_index].detach().cpu().numpy() if isinstance(self.inferred_intents[ai_index], torch.Tensor) else self.inferred_intents[ai_index]
                            idx = np.argmin(np.abs(label_keys - inferred_intent))
                            if self.get_label_dict()[label_keys[idx]] == "right":  # Ado is to the left of ego car.
                                self._eval_metric_helpers["intent_specific_reward"] -= self.config.left_right_bias_penalty
                                if self.config.use_intent_specific_rewards_in_shaping:
                                    left_right_bias_penalty += self.config.left_right_bias_penalty
                        if self.config.use_left_bias_penalty:  # Ado is to the left of ego car.
                            left_right_bias_penalty += self.config.left_right_bias_penalty
                    elif rel_pos_ego_frame[1] < 0:
                        label_keys = np.array(list(self.get_label_dict().keys()))
                        if label_keys.size != 0:
                            ai_index = 0
                            inferred_intent = self.inferred_intents[ai_index].detach().cpu().numpy() if isinstance(self.inferred_intents[ai_index], torch.Tensor) else self.inferred_intents[ai_index]
                            idx = np.argmin(np.abs(label_keys - inferred_intent))
                            if self.get_label_dict()[label_keys[idx]] == "left":  # Ado is to the right of ego car.
                                self._eval_metric_helpers["intent_specific_reward"] -= self.config.left_right_bias_penalty
                            if self.config.use_intent_specific_rewards_in_shaping:
                                left_right_bias_penalty += self.config.left_right_bias_penalty
                        if self.config.use_right_bias_penalty:  # Ado is to the right of ego car.
                            left_right_bias_penalty += self.config.left_right_bias_penalty
                    break  # No need to check other ados if we're already within distance.

                # Additional metrics for wandb logging
                if self.ai_action_index is not None:
                    ai_steering_action = acts[self.ai_action_index][STEER_INDEX]

                    if dist < self.config.ego_distance_threshold:
                        rel_pos_ego_frame = self.get_rel_pos_ego_frame(ado_position, ego_position, ego_car)
                        if rel_pos_ego_frame[1] > 0 and ai_steering_action < 0:
                            # Ado is to the left of ego car, but still ego steered left
                            self.incorrect_left_steer_count += 1
                        elif rel_pos_ego_frame[1] < 0:
                            # Ado is to the right of ego car, but still ego steered right
                            self.incorrect_right_steer_count += 1

                    if dist < self.config.ego_collision_course_distance_threshold:
                        if abs(ai_steering_action) < self.config.ego_collision_course_steering_angle_radian_threshold:
                            self.incorrect_steering_angle_count += 1

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

        # === Soft bound penalty for veering too close to the edge of the track ===
        soft_bound_penalty = 0.
        if self.config.use_soft_bounds_penalty:
            # Softly penalize if ego goes too close to the edge of the track
            ego_distances = np.linalg.norm(np.array(track)[:, 2:4] - np.array(ego_car.hull.position), axis=1)
            ego_track_index = np.argmin(ego_distances)
            # TODO(jon): Add documentation explaining why the minus pi is needed here.
            left_bound = ego_car.hull.position - np.array(track_left[ego_track_index])
            right_bound = ego_car.hull.position - np.array(track_right[ego_track_index])
            dist_to_left_bound = np.linalg.norm(left_bound)
            dist_to_right_bound = np.linalg.norm(right_bound)
            if min(dist_to_left_bound, dist_to_right_bound) < self.config.soft_bounds_penalty_threshold and bound_penalty == 0.0:
                soft_bound_penalty += self.config.soft_bounds_penalty_multiplier * (
                    1.0 - min(dist_to_left_bound, dist_to_right_bound) / self.config.soft_bounds_penalty_threshold
                )

        # === Penalties governing human or AI actuation smoothness ===
        # N.B. These were especially necessary in preventing the AI agent's steering interventions from being too noisy and jerky.
        action_smoothness_penalty = 0.
        steering_difference = 0.
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
                    if self._is_verbose:
                        print(f"  steering penalty multiplier {multiplier}")
                action_smoothness_penalty = multiplier * np.linalg.norm(steering_difference)
        self._last_acts_for_reward_shaping = acts

        # === Penalties for extremely low speeds ===
        speed_penalty = 0.
        if track_rel_vel[0] < 0.5 and not within_inner_coll_dist:
            speed_penalty = self.config.default_low_speed_penalty
            if self._use_speed_penalty:
                speed_penalty = self.config.additional_low_speed_penalty

        # === Penalty for high speeds near ado ===
        if self.config.use_near_ado_speed_penalty and within_near_ado_dist:
            speed_penalty += self.config.near_ado_speed_penalty_multiplier * track_rel_vel[0]

        # === Bonus for higher speeds ===
        speed_bonus = 0.
        if self.config.use_speed_bonus:
            if self._use_collision_penalty:
                if not within_outer_coll_dist:
                    speed_bonus = self.config.speed_bonus_multiplier * track_rel_vel[0]
            elif not within_inner_coll_dist:
                speed_bonus = self.config.speed_bonus_multiplier * track_rel_vel[0]

        # === Combine all the rewards and penalties ===
        bonuses_and_penalties = speed_bonus - (action_smoothness_penalty + bound_penalty + soft_bound_penalty + collision_penalty + speed_penalty + left_right_bias_penalty)
        if self._is_verbose:
            print(f"reward: {reward}, bonuses_and_penalties: {bonuses_and_penalties} (speed_bonus: {speed_bonus}, action_smoothness_penalty: {action_smoothness_penalty}, bound_penalty: {bound_penalty}, soft_bound_penalty: {soft_bound_penalty}, collision_penalty: {collision_penalty}, speed_penalty: {speed_penalty}), left_right_bias_penalty: {left_right_bias_penalty})")
        self._eval_metric_helpers["reward_components"] = {
            "base_reward": reward,
            "speed_bonus": speed_bonus * reward_scalar,
            "action_smoothness_penalty": -action_smoothness_penalty * reward_scalar,
            "bound_penalty": -bound_penalty * reward_scalar,
            "soft_bound_penalty": -soft_bound_penalty * reward_scalar,
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

        self.reward_component_names = list(self._eval_metric_helpers["reward_components"].keys())  # Store the keys separately, since we have to pass the rewards as a list later on.
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
        vehicle_action = self.allocate_actions(policy_action)  # Allocate the actions to the underlying ego vehicle.
        collapsed_actions = self.collapse_actions(
            policy_action
        )  # Collapse acceleration / deceleration to a single accel value.

        total_reward = [0.0] * self.num_agents

        for repeat in range(self._repeat):
            obs, reward_item, over, info_item = self.base_env.step(vehicle_action)

            # Boost the base driving reward to more strongly incentivize driving.
            reward = reward_item * 100.0

            info = info_item

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

        obs["image"] = np.array([[[0, 0, 0]]], dtype=np.uint8)

        # Replicate the observation for all agents and append the action of the other agents
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
            # Note: We assume capacity to observe one other egocentric agent; however, we don't have access to that agent, so we just pad the actions with zeros.
            obs["state"] = [
                np.concatenate([obs["state"], np.zeros_like(actions[0])])
            ]  # Append a null action to enable eventual use in a two-egocentric agent env.

        self._done = over or (self._length and self._step >= self._length)

        if self.ai_action_index is not None:
            self.log_metrics(actions[self.ai_action_index])
        else:
            self.log_metrics([0.0, 0.0])  # No AI actions, so log a null action.
        # TODO(jon) Re-introduce metrics / info such as collision and action discontinuity below.
        info.update(
            {
                "speed": self._eval_metric_helpers["current_speed"][-1] if self.ai_action_index is not None else 0,
                "intervention_norm": (
                    self._eval_metric_helpers["current_intervention_norm"][-1]
                    if self.ai_action_index is not None
                    else 0
                ),
                "progress": self.base_env.tile_visited_count[self.ego_index] / len(self.base_env.track) if self.base_env.track is not None else 0,
                "collisions": self._eval_metric_helpers["collisions"],
                "intent_specific_reward": self._eval_metric_helpers["intent_specific_reward"],
                "reward_components": list(self._eval_metric_helpers["reward_components"].values()),
            }
        )

        obs["is_terminal"] = False if over is None else over
        obs["is_first"] = False
        done = False if over is None else over

        return obs, total_reward, done, info

    def reset(self):
        self._done = False
        self._step = 0

        # This has to happen before the env reset, otherwise reset will populate the ado cars incorrectly.
        if hasattr(self.config, "ado_rails"):
            if self.config.use_demo_ado_rails:
                self.reset_demo_ado_rails()  # Populates self._ref_ado_csv and self._base_ado_timestep.
            else:
                self.reset_ado_rails()  # Populates self._ref_ado_csv and self._base_ado_timestep.
            self.base_env.reset_ado_rails(
                ado_rails=self._ref_ado_csv,
                ado_trigger_groups=self._ado_trigger_groups,
                ado_speed_factors=[self.make_random_speed_factor() for _ in range(len(self._ado_trigger_groups))],
            )

        result = self.base_env.reset()
        state = result[0]["state"]  # For agent 0
        obs = result[0]

        self.last_ego_position = [self.base_env.ego_car.hull.position.x, self.base_env.ego_car.hull.position.y]

        obs = {
            "image": np.array([[[0, 0, 0]]], dtype=np.uint8),
            "state": state,
            "is_terminal": np.array(False),
            "is_first": np.array(True),
        }
        self._record_and_reset_eval_tracking()
        self._reset_lap_progress()

        # obs = {key: [val] if len(val.shape) == 0 else val for key, val in obs.items()}
        obs["is_terminal"] = False
        obs["is_first"] = True

        # Assume the first action is zero; use this as the observation.  These should correspond to a resonable null
        # action (e.g. initial condition of the Box2d Car model, default (null) AI input).
        default_action = self.init_actions()
        obs["state"] = [
            np.concatenate([obs["state"], default_action[self.ego_index]]) for _ in range(self._num_egocentric_agents)
        ]

        self.road_poly = self.base_env.road_poly

        return obs

    def close(self):
        return self.base_env.close()
