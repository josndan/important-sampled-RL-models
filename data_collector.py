from experiment import Simulator
from utils import CustomDefaultDict, normalize
from collections import defaultdict
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

    def get_correction_policy(self, pi, estimate_separately=False, without_correction_term=False):
        if self.history is None:
            raise Exception("Data not yet collected")

        d_0_obs = defaultdict(float)
        estimated_denominator = CustomDefaultDict(self.world.actions, CustomDefaultDict(self.world.observations, 0))
        for state_local in self.world.states:
            for obs in self.world.observations:
                for action in self.data_collecting_agent.actions:
                    estimated_denominator[action][obs] += self.estimate([state_local], [obs], False) * \
                                            self.data_collecting_agent.policy[state_local][action]
                d_0_obs[obs] += self.world.observation_function[state_local][obs] \
                                       * self.estimate_based_on_t(state_local, 0)


        policy = CustomDefaultDict(self.world.states, CustomDefaultDict(self.world.actions, 0))

        for state_from_mu in (self.world.states - self.world.absorbing_states):
            for action in self.data_collecting_agent.actions:
                for obs in self.world.observations:
                    if without_correction_term:
                        policy[state_from_mu][action] += self.world.observation_function[state_from_mu][obs] \
                                                         * pi[obs][action]
                    else:
                        estimated_numerator = self.estimate([state_from_mu], [obs], False)

                        policy[state_from_mu][action] += d_0_obs[obs] * pi[obs][action] \
                                                         * self.data_collecting_agent.policy[state_from_mu][action] \
                                                         * estimated_numerator / estimated_denominator[action][obs]

                        # d_0_obs_temp = self.estimate_based_on_t(obs, 0)
                        # if d_0_obs_temp != d_0_obs[obs]:
                        #     print(d_0_obs_temp,  d_0_obs[obs])
                        #     raise Exception()

                d_0_state = self.estimate_based_on_t(state_from_mu, 0)
                policy[state_from_mu][action] /= d_0_state

            # policy[state_from_mu] = normalize(policy[state_from_mu])

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

    def estimate_based_on_t(self, qtn, t):
        if (qtn, t) in self.estimated_cache:
            return self.estimated_cache[(qtn, t)]

        numerator = 0
        for episode in self.history:
            episode_str = ''.join(episode[t])
            numerator += qtn in episode_str
        self.estimated_cache[(qtn, t)] = numerator / len(self.history)

        return self.estimated_cache[(qtn, t)]

    def collect(self):
        simulator = Simulator(self.world, False)
        self.history = [simulator.run(self.data_collecting_agent, 0)[0] for _ in range(self.num_epi)]
