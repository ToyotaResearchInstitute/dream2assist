import numpy as np
import torch


class Shaper:
    def __init__(
        self,
        human_inference_reward_type,
        egocentric_agent_names,
        human_agents,
        use_intent,
        frozen_agent_id=None,
        indices={"AI": 0, "human": 1},
        ai_acts_penalty_multiplier=1.0,
    ):
        """
        Initialization for the Shaper class, which shapes the reward for the AI agent based on human alignment.

        Parmameters:
            human_inference_reward_type (str): Type of reward for human inference ('action', 'action_logprob', or 'reward').
            egocentric_agent_names (list): List of egocentric agent names.
            human_agents (list): List of human agents.
            use_intent (bool): Whether to use intent in shaping the reward.
            frozen_agent_id (int, optional): ID of the agent that is frozen and not updated during training. Defaults to None.
            indices (dict): Dictionary mapping agent names to their indices.
            ai_acts_penalty_multiplier (float): Multiplier for the AI agent's action penalty. Defaults to 1.0.
        """
        self._human_inference_reward_type = human_inference_reward_type
        self._num_egocentric_agents = len(egocentric_agent_names)
        self._use_intent = use_intent
        self._egocentric_agent_names = egocentric_agent_names
        self._frozen_agent_id = frozen_agent_id
        self._human_agents = human_agents
        self._ai_acts_penalty_multiplier = ai_acts_penalty_multiplier

        self._label = None
        self._inferred_intents = None
        self._last_inferred_intent = None
        self._human_agent_state = (
            None  # Note: be careful to reset this if an agent's model is switched due to inference.
        )
        self._indices = indices

    def set_inferred_intents(self, intents):
        """
        Sets the inferred intents for the AI agent.

        Parameters:
            intents (list): List of inferred intents for the AI agent.
        """
        print("Setting inferred intents", intents)
        self._inferred_intents = intents
        self._last_inferred_intent = None

    def form_human_observation(self, obs, actions, over=None):
        # We need to make an explicit obs to allow shape_rewards to evaluate optimal actions and rewards from the optimal inferred human model.
        # The obs from the one we use below because it does not contain rewards.  That, of course, will be populated by shape_rewards;
        # however, we don't need these for evaluating the human model, permitting us to zero these out for purposes of human model evaluation.
        obs_struct_no_reward = [
            {
                # We are not using the image field, so we initialize to a dummy shape of [,,3]
                # to match the expected dimensions of an RGB image.
                "image": np.array([[[0, 0, 0]]], dtype=np.uint8),
                "state": obs,
                "is_terminal": np.array(False) if over is None else np.array(over),
                "is_first": np.array(False),
                # The reward consists of [human, AI] rewards and will always be 1 x 2.
                "reward": [0.0, 0.0],
            }
        ]
        # Append the action of the other agents
        if self._num_egocentric_agents > 1:
            obs_struct_no_reward[0]["state"] = [obs_struct_no_reward[0]["state"]]
            if self._frozen_agent_id is not None:
                # Zero out the actions of the frozen agent, as observed by the unfrozen agent.
                unfrozen_agent_id = 1 if self._frozen_agent_id == 0 else 0
                obs_struct_no_reward[0]["state"][unfrozen_agent_id][-2:] = 0.0
        else:  # Only one agent.
            # Note: We assume capacity to observe one other agent; however, we don't have access to that agent, so we just pad the actions with zeros.
            obs_struct_no_reward[0]["state"] = [
                np.concatenate([obs_struct_no_reward[0]["state"], np.zeros_like(actions[0])])
            ]  # Append a null action to enable eventual use in a two-egocentric agent env.

        obs_no_reward = {}
        for k in obs_struct_no_reward[0]:
            obs_no_reward[k] = (
                [o[k] for o in obs_struct_no_reward]
                if isinstance(obs_struct_no_reward[0][k], tuple)
                else np.stack([o[k] for o in obs_struct_no_reward])
            )
        return obs_no_reward

    def shape_rewards(self, reward, obs, acts):
        """
        Shapes the reward for the AI and human, given actions and observations, and other meta-info of the optimal human agent (optimal action, reward).

        Parameters:
            reward (list): The reward from the environment.
            obs (dict): The observation from the environment.
            acts (dict): The actions from the environment.

        Returns:
            list: A list of rewards for the AI and human.
        """
        if self._num_egocentric_agents == 1:
            # If only one ego agent, that agent assumes full control, so we don't penalize actions.
            return reward, {}
        elif self._num_egocentric_agents != 2:
            raise ValueError(f"Unsupported number of agents: {self._num_egocentric_agents}")

        acts = np.stack(acts["action"])
        print("Shape rewards acts", acts)
        ai_acts_penalty = self._ai_acts_penalty_multiplier * np.linalg.norm(acts.take(self._indices["AI"]))  # Soft penalty on AI actions.

        ai_match_bonus = 0.0
        result = reward.copy()
        if self._human_agents is not None and len(self._human_agents) > 0 and self._use_intent:
            assert (
                self._inferred_intents is not None
            )  # This needs to be set external to this env, since inference is done at the policy level.
            inferred_intent = self._inferred_intents[self._indices["AI"]]
            if self._last_inferred_intent != inferred_intent:
                self._human_agent_state = None
            self._last_inferred_intent = (
                inferred_intent.detach().cpu().numpy() if isinstance(inferred_intent, torch.Tensor) else inferred_intent
            )
            done = np.array([False])
            # TODO(jon): Generalize, and update if going with an integer/one-hot intent vector.
            if not isinstance(inferred_intent, float) and not isinstance(inferred_intent, int):
                intent_id = np.argmax(inferred_intent.detach().cpu().numpy())
            else:
                intent_id = int(inferred_intent > 0.5)
            obs_no_reward = self.form_human_observation(obs["state"][0], acts)
            opt_human_action, self._human_agent_state = self._human_agents[intent_id](
                obs_no_reward, done, self._human_agent_state, training=False
            )
            if self._human_inference_reward_type == "action":
                # Reward for AI's action nudging the human to reach their inferred optimal action.
                ai_match_bonus += -np.linalg.norm(
                    acts.take(self._indices["human"])
                    + acts.take(self._indices["AI"])
                    - opt_human_action["action"][0][0].detach().cpu().numpy()
                )
            elif self._human_inference_reward_type == "action_logprob":
                act_with_ai = acts.take(self._indices["human"]) + acts.take(self._indices["AI"])
                act_with_ai = torch.tensor(act_with_ai).to("cuda:0")
                label = self._label if self._label is not None else 0
                log_prob = self._human_agents[label].action_log_prob(
                    act_with_ai, obs, done, states=self._human_agent_state
                )
                # TODO replace hard-coded indices with variables.
                log_prob = log_prob[0][0]  # Isolate the human agent's log-probability.
                # Reward for AI's action nudging the human to reach actions consistent with high-likelihood ground-truth (oracular) human actions.
                ai_match_bonus += log_prob.detach().cpu().numpy()
            if self._human_inference_reward_type == "reward" or self._human_inference_reward_type == "action":
                opt_human_reward = self._human_agents[intent_id].get_reward([obs_no_reward], [self._human_agent_state])
                # Reward for AI's action nudging the human to reach their inferred optimal action.
                # TODO replace hard-coded indices with variables.
                ai_match_bonus += opt_human_reward[0][0][0].detach().cpu().numpy()
            else:
                raise ValueError(f"Unsupported human inference reward type: {self._human_inference_reward_type}")
        # human_acts_reward = np.linalg.norm(acts[self._indices["human"]] - acts[self._indices["AI"]])  # Reward for human actions.
        result[self._indices["AI"]] += -ai_acts_penalty + ai_match_bonus
        print(f"Reward shaping: base {reward}, ai_penalty {-ai_acts_penalty}, ai_bonus {ai_match_bonus}, result {result}")
        components = {
            "ai_action_penalty": -ai_acts_penalty,
            "ai_human_alignment_bonus": ai_match_bonus,
        }
        return result, components
