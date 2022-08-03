from utils import get_random, CustomDefaultDict, validate_prob_axiom


class Agent:
    def __init__(self, states_or_observations, actions):
        self.policy = CustomDefaultDict(states_or_observations, CustomDefaultDict(actions, 0))
        self.states_or_observations = states_or_observations
        self.actions = set()

    def update_actions(self):
        for state_or_obs in self.policy:
            validate_prob_axiom(self.policy[state_or_obs])
            self.actions.update([action for action in self.policy[state_or_obs] if self.policy[state_or_obs][action]])

    def initialize_policy(self, policy):
        self.policy.update(policy)
        print(policy)
        self.update_actions()

    def get_action(self, state):
        return get_random(self.policy[state])


class AgentOnObservation(Agent):
    def __init__(self, *args, **kwargs):
        super(AgentOnObservation, self).__init__(*args, **kwargs)
