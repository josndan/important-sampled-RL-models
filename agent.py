from utils import get_random, CustomDefaultDict


class Agent:
    def __init__(self, states_or_observations, actions):
        self.policy = CustomDefaultDict(states_or_observations, CustomDefaultDict(actions, 0))
        self.states_or_observations = states_or_observations

    def initialize_policy(self, policy):
        self.policy.update(policy)

    def get_action(self, state):
        return get_random(self.policy[state])


class AgentOnObservation(Agent):
    def __init__(self, *args, **kwargs):
        super(AgentOnObservation, self).__init__(*args, **kwargs)
