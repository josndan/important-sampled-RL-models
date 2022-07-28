from utils import get_random


class Agent:
    def __init__(self, policy={}):
        self.policy = policy

    def get_action(self, state):
        return get_random(self.policy[state])


class AgentOnObservation(Agent):
    def __init__(self, *args, **kwargs):
        super(AgentOnObservation, self).__init__(*args, **kwargs)
