from experiment import Simulator
from utils import CustomDefaultDict, normalize, validate_prob_axiom
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

    def get_correction_policy(self, pi, without_correction_term=False, debug=False, validate_everything=True):
        if self.history is None:
            raise Exception("Data not yet collected")

        d_0_obs = CustomDefaultDict(self.world.observations, 0)
        d_0_state = CustomDefaultDict(self.world.states, 0)
        estimated_denominator = CustomDefaultDict(self.world.observations, CustomDefaultDict(self.world.actions, 0))
        estimated_numerator = CustomDefaultDict(self.world.observations, CustomDefaultDict(self.world.actions, 0))
        bias = CustomDefaultDict(self.world.observations,
                                 CustomDefaultDict(self.world.actions, CustomDefaultDict(self.world.states, 0)))

        for obs in self.world.observations:
            for state_local in (self.world.states - self.world.absorbing_states):
                for action in self.data_collecting_agent.actions:
                    estimated_denominator[obs][action] += self.estimate([state_local], [obs], False) * \
                                                          self.data_collecting_agent.policy[state_local][action]
                d_0_obs[obs] += self.world.observation_function[state_local][obs] \
                                * self.estimate_based_on_t(state_local, 0)
                estimated_numerator[obs][state_local] = self.estimate([state_local], [obs], False)

                d_0_state[state_local] = self.estimate_based_on_t(state_local, 0)

            validate_prob_axiom(estimated_numerator[obs])
            validate_prob_axiom(estimated_denominator[obs])

        validate_prob_axiom(d_0_obs)
        validate_prob_axiom(d_0_state)

        policy = CustomDefaultDict(self.world.states, CustomDefaultDict(self.world.actions, 0))

        if validate_everything:
            if debug:
                print("Verifying observation")
            validate_prob_axiom(d_0_obs)
            if debug:
                print("Verifying states")
            validate_prob_axiom(d_0_state)

        error = 0

        for state_from_mu in (self.world.states - self.world.absorbing_states):
            for action in self.data_collecting_agent.actions:
                for obs in self.world.observations:
                    if without_correction_term:
                        policy[state_from_mu][action] += self.world.observation_function[state_from_mu][obs] \
                                                         * pi[obs][action]
                    else:
                        bias[obs][action][state_from_mu] = self.data_collecting_agent.policy[state_from_mu][action] \
                                                           * estimated_numerator[obs][state_from_mu] / \
                                                           estimated_denominator[obs][action]

                        # bias[obs][action][state_from_mu] = self.estimate([state_from_mu], [obs, action], False)
                        policy[state_from_mu][action] += d_0_obs[obs] * pi[obs][action] * bias[obs][action][
                            state_from_mu]
                if not without_correction_term:
                    policy[state_from_mu][action] /= d_0_state[state_from_mu]

            # policy[state_from_mu] = normalize(policy[state_from_mu])
        if debug:
            print(bias)
            for action in self.data_collecting_agent.actions:
                for obs in self.world.observations:
                    validate_prob_axiom(bias[obs][action])
                    error += abs(1 - sum(bias[obs][action].values()))

            print()
            print("Total error", error)
            print()
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
