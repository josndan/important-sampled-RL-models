from experiment import Simulator
from utils import get_random, CustomDefaultDict, validate_prob_axiom


class MDP:
    def __init__(self, display_history=False):
        self.states = set()
        self.actions = set()  # TODO find out if actions is required. It's currently not used
        self.absorbing_states = set()
        self.display_history = display_history
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

    def __calculate_absorbing(self):
        for state in self.states:
            if all([all([y == 0 for y in x.values()]) for x in self.transition[state].values()]):
                self.absorbing_states.add(state)
                for action in self.actions:
                    self.transition[state][action][state] = 1

    def validate_transition(self):
        for state in self.transition.keys():
            for x in self.transition[state].values():
                validate_prob_axiom(x)

    def initialize_world(self, states, transition, rewards, initial_dist, actions, *args, **kwargs):

        self.states.update(states)
        self.actions.update(actions)
        self.transition.update(transition)
        # self.absorbing_states = self.states - set(self.transition.keys())
        self.__calculate_absorbing()
        self.rewards.update(rewards)
        self.initial_dist.update(initial_dist)

        validate_prob_axiom(self.initial_dist)
        self.validate_transition()

        if not isinstance(self, POMDP):  # Because for POMDP it calls POMDP.reset instead of MDP.reset
            self.reset()

    def reset(self):
        self.current_state = get_random(self.initial_dist)
        if self.display_history:
            print("\n")

    def take_action(self, action):
        if self.display_history:
            print(f"{self.current_state}, ", end="")
        ret = self.rewards[self.current_state, action]
        self.current_state = get_random(self.transition[self.current_state][action])
        if self.display_history:
            print(f"{action}, {ret}; ", end="")
        return ret

    def get_current_state(self):
        return self.current_state

    def reached_absorbing(self):
        return self.current_state in self.absorbing_states


class POMDP(MDP):
    def __init__(self, *args, **kwargs):
        super(POMDP, self).__init__(*args, **kwargs)
        self.observations = set()
        self.current_observation = ""
        self.observation_function = CustomDefaultDict(self.states, CustomDefaultDict(
            self.observations, 0
        ))

    def initialize_world(self, states, transition, rewards, initial_dist, actions, observations, observation_function,
                         *args,
                         **kwargs):
        super(POMDP, self).initialize_world(states, transition, rewards, initial_dist, actions, *args, **kwargs)
        self.observations.update(observations)
        self.observation_function.update(observation_function)
        self.reset()

    def take_action(self, action):
        if self.display_history:
            print(f"{self.current_observation}, ", end="")
        reward = super(POMDP, self).take_action(action)
        self.current_observation = get_random(self.observation_function[self.current_state])
        return reward

    def get_current_observation(self):
        return self.current_observation

    def reset(self):
        super(POMDP, self).reset()
        self.current_observation = get_random(self.observation_function[self.current_state])


# Pre Condition: data collecting policy is a policy on states
class DataCollector:
    def __init__(self, world, data_collecting_policy):
        self.history = None
        self.world = world
        self.data_collecting_policy = data_collecting_policy
        self.estimated_cache = {}

    def get_correction_policy(self, pi, estimate_separately=False):
        if self.history is None:
            raise Exception("Data not yet collected")

        policy = CustomDefaultDict(self.world.states, CustomDefaultDict(self.world.actions, 0))
        for state_from_mu in self.world.states:
            for action in self.world.actions:
                for state in self.world.states:
                    for obs in self.world.observations:

                        estimated_numerator = self.estimate([state_from_mu], [obs])

                        if estimate_separately:
                            estimated_denominator = self.estimate([action], [obs])
                        else:
                            estimated_denominator = 0
                            for state_local in self.world.states:
                                estimated_denominator += self.estimate([state_local], [obs]) * \
                                                         self.data_collecting_policy[state_local][action]

                        policy[state_from_mu][action] += self.world.observation_function[state][obs] * pi[obs][action] \
                                                      * self.data_collecting_policy[state_from_mu][action] \
                                                      * estimated_numerator / estimated_denominator

        return policy

    # probability_of and given are lists
    # each element in the list must be a valid state or observation or action name
    def estimate(self, probability_of, given):

        if (tuple(probability_of), tuple(given)) in self.estimated_cache:
            return self.estimated_cache[(tuple(probability_of), tuple(given))]

        numerator = 0
        denominator = 0
        intersection = set(probability_of + given)
        given_set = set(given)
        for episode in self.history:
            numerator += intersection <= set(episode)
            denominator += given_set <= set(episode)

        if denominator == 0:
            raise Exception("Number of given instances in episode is 0")

        self.estimated_cache[(tuple(probability_of), tuple(given))] = numerator / denominator

        return numerator / denominator

    def collect(self):
        simulator = Simulator(self.world, False)
        self.history = simulator.run(self.data_collecting_policy)[0]
