from utils import get_random


class Agent:
    def __init__(self, policy={}):
        self.policy = policy

    def get_action(self, state):
        return get_random(self.policy[state])
