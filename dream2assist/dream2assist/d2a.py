import os
import pathlib
import sys

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import wandb

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools

import torch
from torch import nn
from torch import distributions as torchd


to_np = lambda x: x.detach().cpu().numpy()


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


class Dream2Assist(nn.Module):
    """Implements a neural network model for handling multiple agents in a simulated environment.

    Attributes:
        Various configuration parameters, logging tools, internal state, and behaviors for each agent.

    Methods:
        __init__: Sets up the configuration, logger, and initializes behaviors for agents.
        load_agent_sub_state_dict: Loads specific parts of a saved state dictionary for a single agent.
        obs_agent: Processes observation for a single agent in multi-agent scenarios.
        data_per_agent: Processes data specifically for a single agent.
        __call__: Handles the model's forward pass, training steps, logging, and managing agent states.
        _policy & _single_policy: Define the policy for the agent's actions in the environment.
        _exploration: Adds exploration noise to the agent's actions.
        _population_train: Handles the training of the population of agents, including training from a checkpoint.
        _train: Conducts the training step for the agents.

    """

    def __init__(self, obs_spaces, act_spaces, configs, logger, dataset=None, offline_dataset=None, names=None):
        """Initializes the Dream2Assist model with given observation spaces, action spaces, configuration, logger,
        dataset, and names.

        Parameters:
            obs_spaces: Observation spaces of the agents.
            act_spaces: Action spaces of the agents.
            configs: A list of configuration settings for the sub-models.
            logger: Logger for recording training and evaluation metrics.
            dataset: Dataset used for training the model. Default is None.
            offline_dataset: Offline-collected dataset for training, e.g., from human experiments. Default is None.
            names: Names of the agents, if training. Default is None.

        """
        super(Dream2Assist, self).__init__()
        self._config = configs[0]
        self._logger = logger
        self._should_log = tools.Every(self._config.log_every)
        batch_steps = self._config.batch_size * self._config.batch_length
        self._should_train = tools.Every(batch_steps / self._config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(self._config.reset_every)
        self._should_expl = tools.Until(int(self._config.expl_until / self._config.action_repeat))
        self._metrics = {}
        self._step = count_steps(self._config.traindir)
        self._update_count = 0
        self._names = names
        # Schedules.
        for i in range(len(configs)):
            configs[i].actor_entropy = lambda x=configs[i].actor_entropy: tools.schedule(x, self._step)
            configs[i].actor_state_entropy = lambda x=configs[i].actor_state_entropy: tools.schedule(x, self._step)
            configs[i].imag_gradient_mix = lambda x=configs[i].imag_gradient_mix: tools.schedule(x, self._step)
        self._dataset = dataset
        self._offline_dataset = offline_dataset
        self._wm = nn.ModuleList([])
        self._task_behavior = nn.ModuleList([])
        self._expl_behavior = nn.ModuleList([])
        self._num_actions = []
        # config.use_intent = True
        assert len(obs_spaces) == len(act_spaces)
        if self._names is not None:
            assert len(obs_spaces) == len(self._names)
        # Initialize sub-models for each agent.
        for obs_space, act_space, config in zip(obs_spaces, act_spaces, configs):
            config.num_actions = act_space.n if hasattr(act_space, "n") else act_space.shape[0]
            self._num_actions.append(config.num_actions)
            wm = models.WorldModel(obs_space, act_space, self._step, config)
            task_behavior = models.ImagBehavior(config, wm, config.behavior_stop_grad)
            if config.compile:
                wm = torch.compile(wm)
                task_behavior = torch.compile(task_behavior)
            self._wm.append(wm)
            self._task_behavior.append(task_behavior)
            reward = lambda f, s, a: wm.heads["reward"](f).mean
            self._expl_behavior.append(
                dict(
                    greedy=lambda: task_behavior,
                    random=lambda: expl.Random(config),
                    plan2explore=lambda: expl.Plan2Explore(config, wm, reward),
                )[config.expl_behavior]().to(self._config.device)
            )
        self._agent_training = [True] * len(act_spaces)
        if self._config.pct_population_play > 0:
            self._agent_training[self._config.frozen_agent_id] = False
        self.data_label = None
        self._use_prior = self._config.use_prior

    def state_screener(self, states, reset):
        """Resets the states of the agents if necessary, while preserving dimensions and datatypes.

        Parameters:
            states: States of the agents.
            reset: Whether the agents should be reset.

        Returns:
            states: The updated states of the agents.
        """
        step = self._step
        if self._should_reset(step):
            states = None
        if states is not None and reset:
            for k in range(len(states)):
                for key in states[k][0].keys():
                    for i in range(states[k][0][key].shape[0]):
                        states[k][0][key][i] *= 0
                for i in range(len(states[k][1])):
                    states[k][1][i] *= 0
        return states

    def get_intents_from_policy_states(self, states):
        """Extracts the intents from the policy states of the agents.

        Parameters:
            states: Policy states of the agents.

        Returns:
            intents_mean: The mean intents of the agents.
        """
        intents_mean = []
        for wm, p in zip(self._wm, states):
            latent = p[0]
            intent = wm.heads["intent"](wm.dynamics.get_feat(latent))
            intents_mean.append(intent.mean.squeeze(0))  # Remove the stray dimensions.
        return intents_mean

    def get_head_outputs(self, obs, states, name):
        """Extracts the outputs of the heads of the agents.

        Parameters:
            obs: Observations of the agents.
            states: States of the agents.
            name: Name of the head to extract.

        Returns:
            outputs: The outputs of the requested head.
        """
        outputs = []
        for s, o in zip(states, obs):
            _, policy_states = self._policy(o, s, training=False)
            sub_output_mean = []
            for wm, p in zip(self._wm, policy_states):
                latent = p[0]
                output = wm.heads[name](wm.dynamics.get_feat(latent))
                if isinstance(output, dict):
                    mean = getattr(output["state"], "mean", None)
                else:
                    mean = getattr(output, "mean", None)
                if callable(mean):
                    sub_output_mean.append(mean().squeeze(0))  # Remove the stray dimensions.
                else:
                    sub_output_mean.append(mean.squeeze(0))  # Remove the stray dimensions.

            outputs.append(torch.stack(sub_output_mean))
        return torch.stack(outputs)

    def get_intent(self, obs, states):
        """Extracts the intents of the agents.

        Parameters:
            obs: Observations of the agents.
            states: States of the agents.

        Returns:
            intents_mean: The mean intents of the agents.
        """
        return self.get_head_outputs(obs, states, "intent")

    def get_decoder_output(self, obs, states):
        """Extracts the decoder outputs of the agents.

        Parameters:
            obs: Observations of the agents.
            states: States of the agents.

        Returns:
            outputs: The decoder outputs of the agents.
        """
        return self.get_head_outputs(obs, states, "decoder")

    def get_reward(self, obs, states):
        """Extracts the rewards of the agents.

        Parameters:
            obs: Observations of the agents.
            states: States of the agents.

        Returns:
            rewards_mean: The mean rewards of the agents.
        """
        return self.get_head_outputs(obs, states, "reward")

    def _action_dist_from_wm(self, obs, state, wm, task_behavior, num_actions):
        """Extracts the action distribution from the world model.

        Parameters:
            obs: Observations of the agents.
            state: State of the agents.
            wm: World model of the agents.
            task_behavior: Task behavior of the agents.
            num_actions: Number of actions available to the agents.

        Returns:
            action_distribution: The action distribution of the agents.
            latent: The latent state of the agents.
        """
        wm.eval()
        task_behavior.eval()
        if state is None:
            batch_size = len(obs["image"])
            latent = wm.dynamics.initial(len(obs["image"]))
            action = torch.zeros((batch_size, num_actions)).to(self._config.device)
        else:
            latent, action = state
        obs = wm.preprocess(obs)
        embed = wm.encoder(obs)
        latent, _ = wm.dynamics.obs_step(latent, action, embed, obs["is_first"], sample=False)
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = wm.dynamics.get_feat(latent)
        action_distribution = task_behavior.actor(feat)
        return action_distribution, latent

    def action_log_prob(self, action_taken, obs, reset, states=None):
        """Calculates the log probabilities of the actions taken by the agents.

        Parameters:
            action_taken: Actions taken by the agents.
            obs: Observations of the agents.
            reset: Whether the agents should be reset.
            states: States of the agents.

        Returns:
            action_log_prob: The log probabilities of the actions taken by the agents.
        """
        # TODO: This currently returns TWO log probs -- one for the AI (agent 0) and one for the "human" (agent 1)
        #  we should eventually clean this up to have "human" agents be independent of AI (load in with no 0th agent)
        states = self.state_screener(states, reset)
        states = [None] * len(self._wm) if states is None else states
        action_log_prob = []
        for i, (state, wm, task_behavior, num_actions) in enumerate(
            zip(
                states,
                self._wm,
                self._task_behavior,
                self._num_actions,
            )
        ):
            action_distribution, _ = self._action_dist_from_wm(
                self.obs_agent(obs, i), state, wm, task_behavior, num_actions
            )
            logprob = action_distribution.log_prob(action_taken)
            action_log_prob.append(logprob)
        return action_log_prob

    def load_agent_sub_state_dict(self, source_agent_id, target_agent_id, state_dict):
        """Loads specific parts of a saved state dictionary from a source agent to update a target agent.

        Parameters:
            source_agent_id: The ID of the source agent to load weights from.
            target_agent_id: The ID of the target agent to populate weights.
            state_dict: The state dictionary to load.
        """
        # Load the state dict for only a sub-agent's parameters.
        wm_prefix = f"_wm.{source_agent_id}."
        tb_prefix = f"_task_behavior.{source_agent_id}."
        eb_prefix = f"_expl_behavior.{source_agent_id}."
        wm_state_dict = {k[len(wm_prefix) :]: v for k, v in state_dict.items() if k.startswith(wm_prefix)}
        tb_state_dict = {k[len(tb_prefix) :]: v for k, v in state_dict.items() if k.startswith(tb_prefix)}
        eb_state_dict = {k[len(eb_prefix) :]: v for k, v in state_dict.items() if k.startswith(eb_prefix)}
        self._wm[target_agent_id].load_state_dict(wm_state_dict)
        self._task_behavior[target_agent_id].load_state_dict(tb_state_dict)
        self._expl_behavior[target_agent_id].load_state_dict(eb_state_dict)

    def obs_agent(self, obs, agent_id):
        """Processes observation for a single agent in multi-agent scenarios.

        Parameters:
            obs: Observations of the agents.
            agent_id: ID of the agent to process the observation for.

        Returns:
            agent_obs: The processed observation for the agent.
        """
        # ***** REMOVE HARD-CODED INDEX *****
        keys = {"state": -2}  # Keys and location of the agent index
        agent_data = lambda v, k: np.take(v, agent_id, axis=keys[k])
        agent_obs = {k: v if k not in keys.keys() else agent_data(v, k) for k, v in obs.items()}
        # TODO Extend support for unequal action sets - we will need to unpad these here.
        return agent_obs

    def data_per_agent(self, dataset, agent_id):
        """Processes data specifically for a single agent.

        Parameters:
            dataset: Dataset to process for the agent.
            agent_id: ID of the agent to process the data for.

        Returns:
            agent_dataset: The processed data for the agent.
        """
        # ***** REMOVE HARD-CODED INDICES *****
        keys = {"action": -2, "logprob": -1}  # Keys and location of the agent index
        agent_data = lambda v, k: np.take(v, agent_id, axis=keys[k])
        agent_dataset = {k: v if k not in keys.keys() else agent_data(v, k) for k, v in dataset.items()}
        # Unpad the action dim
        slc = [slice(None)] * agent_dataset["action"].ndim
        slc[-1] = list(range(self._num_actions[agent_id]))
        agent_dataset["action"] = agent_dataset["action"][tuple(slc)]
        agent_dataset["state"] = self.obs_agent({"state": agent_dataset["state"]}, agent_id)["state"]
        agent_dataset["reward"] = np.take(agent_dataset["reward"], agent_id, axis=-1)
        return agent_dataset

    def __call__(self, obs, reset, states=None, reward=None, training=True):
        """Handles the model's forward pass, training steps, logging, and managing agent states.

        Parameters:
            obs: Observations of the agents.
            reset: Whether the agents should be reset.
            states: States of the agents.
            reward: Reward for the agents.
            training: Whether the agents are training or not.

        Returns:
            policy_outputs: The actions and log probabilities of the agents.
            states: The updated states of the agents.
        """
        step = self._step
        if self._should_reset(step):
            states = None
        if states is not None and reset.any():
            mask = 1 - reset
            for k in range(len(states)):
                for key in states[k][0].keys():
                    for i in range(states[k][0][key].shape[0]):
                        states[k][0][key][i] *= mask[i]
                for i in range(len(states[k][1])):
                    states[k][1][i] *= mask[i]
        if training:
            # TODO (jon) Disabling freezing / unfreezing the population agents for now.
            # logdir = pathlib.Path(self._config.logdir).expanduser()
            pop_checkpoints = None  # [fname for fname in glob.glob(str(logdir / "*checkpoint*.pt"))]
            steps = self._config.pretrain if self._should_pretrain() else self._should_train(step)
            for _ in range(steps):
                data = next(self._dataset)
                offline_data = next(self._offline_dataset) if self._offline_dataset is not None else None
                # self._population_train(data, offline_data, pop_checkpoints)
                # Even if we have data from an earlier human checkpoint, we don't want take training steps with that agent.
                self._train(data, offline_data, self._agent_training)
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    wandb.log({name: float(np.mean(values))})
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    dataset = next(self._dataset)
                    for i, (wm, name) in enumerate(zip(self._wm, self._names)):
                        openl = wm.video_pred(self.data_per_agent(dataset, i))
                        self._logger.video(f"train_openl_{name}", to_np(openl))
                        # pass array of dim (batch, time, channel, width, height)
                        wandb.log({"train_openl": wandb.Video(np.moveaxis(to_np(openl), -1, -3))})
                self._logger.write(fps=True)

        # Assume agents have the same observation space.
        # TODO To generalize this, we'll need to make similar splitting of image / state observations as is done with
        # actions, including splitting the dataset to get fed to the appropriate underlying agent networks.
        policy_outputs, states = self._policy(obs, states, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_outputs, states

    def _policy(self, obs, states_in, training):
        """Defines the policy for the agent's actions in the environment.

        Parameters:
            obs: Observation data for the agents.
            states_in: States of the agents.
            training: Whether the agents are training or not.

        Returns:
            policy_outputs: The actions and log probabilities of the agents.
            states: The updated states of the agents.
        """
        states_in = [None] * len(self._wm) if states_in is None else states_in
        policy_outputs = []
        states = []
        for i, (state, wm, task_behavior, expl_behavior, num_actions) in enumerate(
            zip(
                states_in,
                self._wm,
                self._task_behavior,
                self._expl_behavior,
                self._num_actions,
            )
        ):
            p, s = self._single_policy(
                self.obs_agent(obs, i),
                state,
                wm.to(self._config.device),
                task_behavior,
                expl_behavior,
                training,
                num_actions,
            )
            policy_outputs.append(p)
            states.append(s)
        # flatten the list of dicts into a single dict
        policy_outputs = {k: [a_agent[k] for a_agent in policy_outputs] for k in policy_outputs[0].keys()}
        policy_outputs = {k: list(zip(*v)) for k, v in policy_outputs.items()}  # envs first
        return policy_outputs, states

    def _single_policy(self, obs, state, wm, task_behavior, expl_behavior, training, num_actions):
        """Defines the policy for a single agent's actions in the environment.

        Parameters:
            obs: Observation data for the agent.
            state: State of the agent.
            wm: World model for the agent.
            task_behavior: Task behavior for the agent.
            expl_behavior: Exploration behavior for the agent.
            training: Whether the agent is training or not.
            num_actions: Number of actions available to the agent.

        Returns:
            policy_output: The action and log probability of the agent.
            state: The updated state of the agent.
        """
        if state is None:
            batch_size = len(obs["image"])
            latent = wm.dynamics.initial(len(obs["image"]))
            action = torch.zeros((batch_size, num_actions)).to(self._config.device)
        else:
            latent, action = state
        obs = wm.preprocess(obs)
        embed = wm.encoder(obs)
        posterior_and_prior = wm.dynamics.obs_step(
            latent, action, embed, obs["is_first"], self._config.collect_dyn_sample
        )
        latent = posterior_and_prior[1] if self._use_prior else posterior_and_prior[0]
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = wm.dynamics.get_feat(latent)
        if not training:
            actor = task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor_dist == "onehot_gumble":
            action = torch.one_hot(torch.argmax(action, dim=-1), num_actions)
        action = self._exploration(action, training, num_actions)

        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _exploration(self, action, training, num_actions):
        """Adds exploration noise to the agent's actions.

        Parameters:
            action: Action to add noise to.
            training: Whether the agent is training or not.
            num_actions: Number of actions available to the agent.

        Returns:
            action: The action with added noise.
        """
        amount = self._config.expl_amount if training else self._config.eval_noise
        if amount == 0:
            return action
        if "onehot" in self._config.actor_dist:
            probs = amount / num_actions + (1 - amount) * action
            return tools.OneHotDist(probs=probs).sample()
        else:
            return torch.clip(torchd.normal.Normal(action, amount).sample(), -1, 1)
        # raise NotImplementedError(self._config.action_noise)

    def _population_train(self, data, offline_data, pop_checkpoints):
        """Handles the training of the population of agents, including training from a checkpoint.

        Parameters:
            data: Data for training the agents.
            offline_data: Offline-collected data for training.
            pop_checkpoints: Checkpoints for training from a frozen agent.
        """
        if not pop_checkpoints or np.random.uniform() > self._config.pct_population_play:
            self._train(data, offline_data, self._agent_training)
            return

        partial_agent_training = self._agent_training
        partial_agent_training[self._config.frozen_agent_id] = False

        # Save the latest model weights
        last_state_dict = self.state_dict()
        # Restore a uniformly-sampled checkpoint and train
        sampled_checkpoint = np.random.choice(pop_checkpoints)
        print(f"  Training from human model checkpoint {sampled_checkpoint}")
        self.load_agent_sub_state_dict(
            self._config.source_agent_id, self._config.frozen_agent_id, torch.load(sampled_checkpoint)
        )
        self._train(data, offline_data, partial_agent_training)
        # Restore the original weights
        self.load_agent_sub_state_dict(self._config.source_agent_id, self._config.frozen_agent_id, last_state_dict)

    def _train(self, data, offline_data, agent_training):
        """Conducts the training step for the agents.

        Parameters:
            data: Data for training the agents.
            offline_data: Offline-collected data for training.
            agent_training: Whether the agents are training or not.
        """
        metrics = {}
        for i, (wm, task_behavior, expl_behavior, name) in enumerate(
            zip(self._wm, self._task_behavior, self._expl_behavior, self._names)
        ):
            if not agent_training[i]:
                continue
            offline_data = self.data_per_agent(offline_data, i) if offline_data else None
            if "label" not in list(data.keys()) and self.data_label is not None:
                data["label"] = self.data_label
            # import IPython; IPython.embed()
            post, context, mets = wm._train(self.data_per_agent(data, i), offline_data)
            if "label" in list(data.keys()):
                self.data_label = data.pop("label", None)
            mets = {f"{k}_{name}": v for k, v in mets.items()}
            metrics.update(mets)
            start = post
            # start['deter'] (16, 64, 512)
            reward = lambda f, s, a: wm.heads["reward"](wm.dynamics.get_feat(s)).mode()
            mets = task_behavior._train(start, reward)[-1]
            mets = {f"{k}_{name}": v for k, v in mets.items()}
            metrics.update(mets)
            if self._config.expl_behavior != "greedy":
                mets = expl_behavior.train(start, context, self.data_per_agent(data, i))[-1]
                metrics.update({f"expl_{key}_{name}": value for key, value in mets.items()})
        for key, value in metrics.items():
            if not key in self._metrics.keys():
                self._metrics[key] = [value]
            else:
                self._metrics[key].append(value)
