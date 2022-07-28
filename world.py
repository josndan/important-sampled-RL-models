from utils import get_random, CustomDefaultDict
from collections import defaultdict


class World:
    def __init__(self):
        self.states = set()
        self.actions = set()  # TODO find out if actions is required. It's currently not used
        self.absorbing_states = set()

        self.transition = CustomDefaultDict(self.states,
                                            CustomDefaultDict(self.actions,
                                                              CustomDefaultDict(
                                                                  self.states, 0
                                                              )
                                                              )
                                            )

        self.rewards = {}  # TODO Design this as dictonary of dictionary (like the others) or dictionary of tuples?
        self.initial_dist = CustomDefaultDict(self.states, 0)
        self.current_state = ""

    # Assumed absorbing states are not explicitly writen in transition function
    def initialize_world(self, states, transition, rewards, initial_dist):
        self.states.update(states)
        self.transition.update(transition)
        self.absorbing_states = self.states - set(self.transition.keys())
        self.rewards.update(rewards)
        self.initial_dist.update(initial_dist)
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
