import copy

from experiment import run
from src.world import MDP
from utils import CustomDefaultDict, normalize, validate_prob_axiom, parallelize
from functools import lru_cache
from collections import defaultdict
from tqdm import tqdm


# Pre Condition: data collecting policy is a policy on states
class DataCollector:
    def __init__(self, world, data_collecting_agent, num_epi=1000, epi_len=1e2):
        self.history = None
        self.world = world
        self.data_collecting_agent = data_collecting_agent
        self.num_epi = num_epi
        self.epi_len = epi_len

    def get_estimated_model(self):
        if self.history is None:
            raise Exception("Data not yet collected")

        observations = self.world.observations
        actions = self.world.actions
        n = len(self.history)
        d_0_obs = CustomDefaultDict(self.world.observations, 0)
        rewards = self.world.rewards

        joint_s_a_s = CustomDefaultDict(observations,
                                        CustomDefaultDict(actions,
                                                          CustomDefaultDict(
                                                              observations, 0
                                                          )
                                                          )
                                        )

        joint_s_a = CustomDefaultDict(observations,
                                      CustomDefaultDict(actions,
                                                        0
                                                        )
                                      )

        joint_s_a_s_r = CustomDefaultDict(observations,
                                          CustomDefaultDict(actions,
                                                            CustomDefaultDict(
                                                                observations,
                                                                defaultdict(int))
                                                            )
                                          )
        rewards = set()
        ob_1s = set()
        ob_2s = set()

        # Assumming self.world is a POMDP
        for trajectory in self.history:
            obs_0 = trajectory[0][1]  # First SOAR in the episode
            d_0_obs[obs_0] += 1 / n
            for i in range(len(trajectory) - 1):
                s_1, o_1, a, r, s_2, o_2 = trajectory[i] + trajectory[i + 1][:2]

                r = int(r[1:])  # assumed rewards are integers
                joint_s_a_s[o_1][a][o_2] += 1
                joint_s_a_s_r[o_1][a][o_2][r] += 1
                joint_s_a[o_1][a] += 1
                rewards.add(r)
                ob_2s.add(o_2)
                ob_1s.add(o_1)

        transition = CustomDefaultDict(observations,
                                       CustomDefaultDict(actions,
                                                         CustomDefaultDict(
                                                             observations,
                                                             0
                                                         )
                                                         ))

        reward_function = CustomDefaultDict(observations,
                                            CustomDefaultDict(actions,
                                                              CustomDefaultDict(
                                                                  observations,
                                                                  CustomDefaultDict(
                                                                      rewards, 0
                                                                  )
                                                              )
                                                              ))

        for ob_1 in ob_1s:
            for a in actions:
                for ob_2 in ob_2s:
                    transition[ob_1][a][ob_2] = joint_s_a_s[ob_1][a][ob_2] / joint_s_a[ob_1][a]
                    for r in rewards:
                        if joint_s_a_s_r[ob_1][a][ob_2][r] != 0:
                            reward_function[ob_1][a][ob_2][r] = joint_s_a_s_r[ob_1][a][ob_2][r] / joint_s_a_s[ob_1][a][
                                ob_2]

        mdp = MDP()
        mdp.initialize_world(observations, transition, reward_function, d_0_obs, actions)
        return lambda: copy.deepcopy(mdp)

    def get_correction_policy(self, pi):
        if self.history is None:
            raise Exception("Data not yet collected")

        policy = CustomDefaultDict(self.world.states, CustomDefaultDict(self.world.actions, 0))

        for state in self.world.states:
            obs = max(self.world.observation_function[state], key=self.world.observation_function[state].get)
            policy[state] = pi[obs]

        return policy

    # def get_correction_policy(self, pi, eqn_number=1, debug=False):
    #     if self.history is None:
    #         raise Exception("Data not yet collected")
    #
    #     d_0_obs = CustomDefaultDict(self.world.observations, 0)
    #     d_0_state = CustomDefaultDict(self.world.states, 0)
    #     estimated_denominator = CustomDefaultDict(self.world.observations, CustomDefaultDict(self.world.actions, 0))
    #     estimated_denominator_test = CustomDefaultDict(self.world.observations,
    #                                                    CustomDefaultDict(self.world.actions, 0))
    #     estimated_numerator = CustomDefaultDict(self.world.observations, CustomDefaultDict(self.world.actions, 0))
    #     bias = CustomDefaultDict(self.world.observations,
    #                              CustomDefaultDict(self.world.actions, CustomDefaultDict(self.world.states, 0)))
    #     bias_test = CustomDefaultDict(self.world.observations,
    #                                   CustomDefaultDict(self.world.actions, CustomDefaultDict(self.world.states, 0)))
    #     policy = CustomDefaultDict(self.world.states, CustomDefaultDict(self.world.actions, 0))
    #
    #     if eqn_number != 1:
    #         for obs in self.world.observations:
    #             for state_local in (self.world.states - self.world.absorbing_states):
    #                 for action in self.data_collecting_agent.actions:
    #                     bias_test[obs][action][state_local] = self.estimate((state_local,), (obs, action))
    #                     estimated_denominator[obs][action] += self.estimate((state_local,), (obs,)) * \
    #                                                           self.data_collecting_agent.policy[state_local][action]
    #                     estimated_denominator_test[obs][action] = self.estimate((action,), (obs,))
    #                 d_0_obs[obs] += self.world.observation_function[state_local][obs] \
    #                                 * self.estimate_based_on_t((state_local,), 0)
    #                 estimated_numerator[obs][state_local] = self.estimate((state_local,), (obs,))
    #
    #                 d_0_state[state_local] = self.estimate_based_on_t((state_local,), 0)
    #
    #             validate_prob_axiom(estimated_numerator[obs])
    #             validate_prob_axiom(estimated_denominator[obs])
    #             validate_prob_axiom(estimated_denominator_test[obs])
    #
    #         if debug:
    #             print("Verifying observation")
    #             print(d_0_obs)
    #         validate_prob_axiom(d_0_obs)
    #         if debug:
    #             print("Verifying states")
    #             print(d_0_state)
    #         validate_prob_axiom(d_0_state)
    #
    #         if debug:
    #             print("Estimated numerator")
    #             print(estimated_numerator)
    #
    #             print("Estimated Denominator")
    #             print(estimated_denominator)
    #
    #             print("Estimated Denominator test")
    #             print(estimated_denominator_test)
    #
    #     error = 0
    #     temp_sum = 0
    #     d_0_state_checker = CustomDefaultDict(self.world.states, 0)
    #
    #     for state_from_mu in (self.world.states - self.world.absorbing_states):
    #         for action in self.data_collecting_agent.actions:
    #             for obs in self.world.observations:
    #                 if eqn_number == 1:
    #                     policy[state_from_mu][action] += self.world.observation_function[state_from_mu][obs] \
    #                                                      * pi[obs][action]
    #                 elif eqn_number == 2:
    #                     bias[obs][action][state_from_mu] = self.data_collecting_agent.policy[state_from_mu][action] \
    #                                                        * estimated_numerator[obs][state_from_mu] / \
    #                                                        estimated_denominator[obs][action]
    #                     for state_local in (self.world.states - self.world.absorbing_states):
    #                         policy[state_from_mu][action] += self.world.observation_function[state_local][obs] \
    #                                                          * pi[obs][action] * bias[obs][action][state_from_mu]
    #                 else:
    #                     bias[obs][action][state_from_mu] = self.data_collecting_agent.policy[state_from_mu][action] \
    #                                                        * estimated_numerator[obs][state_from_mu] / \
    #                                                        estimated_denominator[obs][action]
    #
    #                     # bias[obs][action][state_from_mu] = self.estimate((state_from_mu,), (obs, action,))
    #                     policy[state_from_mu][action] += d_0_obs[obs] * pi[obs][action] * bias_test[obs][action][
    #                         state_from_mu]
    #             temp_sum += policy[state_from_mu][action]
    #             d_0_state_checker[state_from_mu] += policy[state_from_mu][action]
    #             if eqn_number == 3:
    #                 policy[state_from_mu][action] /= d_0_state[state_from_mu]
    #
    #         # policy[state_from_mu] = normalize(policy[state_from_mu])
    #
    #     if debug:
    #         print("temp_sum", temp_sum)
    #         print("d_0 from bias")
    #         print(d_0_state_checker)
    #
    #         print("Bias")
    #         print(bias)
    #
    #         print("Bias test")
    #         print(bias_test)
    #         # for action in self.data_collecting_agent.actions:
    #         #     for obs in self.world.observations:
    #         #         validate_prob_axiom(bias[obs][action])
    #         #         error += abs(1 - sum(bias[obs][action].values()))
    #         #
    #         # print()
    #         # print("Total error", error)
    #         # print()
    #     return policy

    # probability_of and given are lists
    # each element in the list must be a valid state or observation or action name
    # order is important, the given and probability_of cannot miss quantities inbetween
    @lru_cache(maxsize=None)
    def estimate(self, probability_of, given):
        # Self-loops in transition cause issues low priority though as this isn't calculated as of now
        numerator = 0
        denominator = 0
        intersection = probability_of + given
        for episode in self.history:
            for event in episode:
                if set(intersection) <= set(event):
                    numerator += 1
                if len(given) != 0 and set(given) <= set(event):
                    denominator += 1

        if len(given) == 0:
            if denominator != 0:
                raise Exception("This shouldn't be happening #sanitycheck")
            denominator = self.epi_len * self.num_epi  # Does not work for MDP with absorbing states

        if denominator == 0:
            raise Exception("Number of given instances in episode is 0")

        return numerator / denominator

    @lru_cache(maxsize=None)
    def estimate_based_on_t(self, qtn, t):
        numerator = 0
        for episode in self.history:
            numerator += set(qtn) <= set(episode[t])

        return numerator / self.num_epi

    def collect(self):
        # self.history = [res[0] for res in
        #                 parallelize(simulator.run, [(self.data_collecting_agent, 0, 1, self.epi_len)] * self.num_epi)]
        print("Collecting Data")
        self.history = [run(self.data_collecting_agent, self.world, 0, 1, self.epi_len, True)[0] for _ in
                        tqdm(range(self.num_epi))]
