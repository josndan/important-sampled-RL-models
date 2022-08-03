from experiment import Simulator
from utils import CustomDefaultDict

from operator import add
from functools import reduce


# Pre Condition: data collecting policy is a policy on states
class DataCollector:
    def __init__(self, world, data_collecting_agent, num_epi=1000):
        self.history = None
        self.world = world
        self.data_collecting_agent = data_collecting_agent
        self.estimated_cache = {}
        self.num_epi = num_epi

    def get_correction_policy(self, pi, estimate_separately=False):
        if self.history is None:
            raise Exception("Data not yet collected")

        policy = CustomDefaultDict(self.world.states, CustomDefaultDict(self.world.actions, 0))
        for state_from_mu in self.world.states:
            for action in self.data_collecting_agent.actions:
                for state in self.world.states:
                    for obs in self.world.observations:

                        estimated_numerator = self.estimate([state_from_mu], [obs], False)

                        if estimate_separately:
                            estimated_denominator = self.estimate([action], [obs])
                        else:
                            estimated_denominator = 0
                            for state_local in self.world.states:
                                estimated_denominator += self.estimate([state_local], [obs], False) * \
                                                         self.data_collecting_agent.policy[state_local][action]
                        policy[state_from_mu][action] += self.world.observation_function[state][obs] * pi[obs][action] \
                                                         * self.data_collecting_agent.policy[state_from_mu][action] \
                                                         * estimated_numerator / estimated_denominator
        return policy

    # probability_of and given are lists
    # each element in the list must be a valid state or observation or action name
    # order is important, the given and probability_of cannot miss quantities inbetween
    def estimate(self, probability_of, given, given_happens_before=True):
        if (tuple(probability_of), tuple(given), given_happens_before) in self.estimated_cache:
            return self.estimated_cache[(tuple(probability_of), tuple(given), given_happens_before)]

        numerator = 0
        denominator = 0
        if given_happens_before:
            intersection = given + probability_of
        else:
            intersection = probability_of + given
        for episode in self.history:
            episode_str = ''.join(reduce(add, episode))
            numerator += episode_str.count(''.join(intersection))
            denominator += episode_str.count(''.join(given))

        if denominator == 0:
            raise Exception("Number of given instances in episode is 0")

        self.estimated_cache[(tuple(probability_of), tuple(given), given_happens_before)] = numerator / denominator

        return numerator / denominator

    def collect(self):
        simulator = Simulator(self.world, False)
        self.history = [simulator.run(self.data_collecting_agent, 0)[0] for _ in range(self.num_epi)]
