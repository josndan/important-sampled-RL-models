from utils import get_random
from collections import defaultdict


class World:
    def __init__(self):
        self.states = set()
        self.actions = set()
        self.absorbing_states = set()

        def states_factory(default):
            def raise_exception():
                raise Exception("Invalid State")
            if dict:
                return lambda state: default if state in self.states else raise_exception()
            else:
                return lambda state: default if state in self.states else raise_exception()

        self.transition = defaultdict(states_factory({}))
        self.rewards = {}
        self.initial_dist = defaultdict(states_factory(0))
        self.current_state = ""

    # Assumed absorbing states are not explicitly writen in transition function
    def initialize_world(self, states, transition, rewards, initial_dist):
        self.states = states
        self.transition.update(transition)
        self.absorbing_states = self.states - set(self.transition.keys())
        self.rewards = rewards
        self.initial_dist = initial_dist
        self.reset()

    def reset(self):
        self.current_state = get_random(self.initial_dist)

    def take_action(self, action):
        ret = self.rewards[self.current_state, action]
        self.current_state = get_random(self.transition[self.current_state][action])
        return ret

    def get_current_state(self):
        return self.current_state

    def reached_absorbing(self):
        return self.current_state in self.absorbing_states
