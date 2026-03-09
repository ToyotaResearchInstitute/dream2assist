import numpy as np


class HumanModelGeneric:
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        pass

    def reset(self):
        return self._env.reset()


class FixedLaggyHumanModel(HumanModelGeneric):
    """
    Implements a fixed lag human model.

    The human agent's action is stored into a FIFO buffer and then retrieved.
    The AI agent's action is unmodified
    """

    def __init__(self, env, config):
        super(FixedLaggyHumanModel, self).__init__(env)
        self.action_buffer_length = config.action_buffer_length
        self.human_action_index = config.egocentric_agent_names.index("human")
        # TODO (deepak.gopinath) assumes that env is EpicDynamicsSharedRacingWrapper because it tries to access action_spaces and not action_space
        # check if there is a more generic way (instead of using EpicDynamicsSharedRacingWrapper specific getters) to access the action_space of the human agent
        self.human_action_buffer = [
            [np.zeros_like(env.action_spaces[self.human_action_index].sample())]
        ] * self.action_buffer_length

    def step(self, action):
        # action - list of length num_egocentric_agents, each element is the (3,) corresponding to the action.shape for each egocentric agent
        # parse human action array and make a list out of it
        human_action = [action[self.human_action_index]]
        # add it to human action buffer
        self.human_action_buffer.append(human_action)
        # retreive lagged human action array
        applied_human_action = self.human_action_buffer.pop(0)  # [(3,)]
        # update the human action index of full action
        action[self.human_action_index] = applied_human_action[0]  # (3,)
        print("Applied human action and action", applied_human_action, action)
        # modify only the human component of action and call step with action
        return self._env.step(action)


class PredictiveHumanModel(HumanModelGeneric):
    """

    Implements a general predictive model of the human. At any time step, if update_human_action is True,
    this wrapper will compute a new human action sequence of length K conditioned on the current env state and AI action.
    Until the human action sequence is read out fully, updates will be suspended. This could be useful for modeling humans who respond to
    "trigger" like events initiated by the AI. This can model human policies of the form p(a_H | a_AI)

    If we need to model p(a_H | a_AI, s_h, then state can be accessed via self.env.get_current_human_state()
    #TODO (deepak.gopinath) need to add this getter in the base env class. For generality we can distinguish human_state vs env_state. For example, human might be conditioning
    their own actions on "noisy" env state or partial env state.
    """

    def __init__(self, env, config):
        super(PredictiveHumanModel, self).__init__(env)
        self.human_future_action_seq_len = config.human_future_action_seq_len
        self.human_future_action_seq = []
        self.human_action_index = config.egocentric_agent_names.index("human")

    def step(self, action):
        ai_action = action[0]
        if len(self.human_future_action_seq) == 0:
            for k in range(self.human_future_action_seq):
                # TODO plan for future human action sequence conditioned on AI action and fill up the human_future_action_seq buffer
                self.human_future_action_seq.append(0)
        human_action = self.human_future_action_seq.pop(0)
        # recompose ai_action and human_action into action
        action[self.human_action_index] = human_action
        return self._env.step(action)
