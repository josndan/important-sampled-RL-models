from utils import get_random, CustomDefaultDict, validate_prob_axiom


class MDP:
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

    def __calculate_absorbing(self):
        for state in self.states:
            if all([all([y == 0 for y in x.values()]) for x in self.transition[state].values()]):
                self.absorbing_states.add(state)
                # for action in self.actions:
                #     self.transition[state][action][state] = 1

    def display(self):
        print("States")
        print(self.states)
        print("Absorbing States")
        print(self.absorbing_states)
        print("Initial Distribution")
        print(self.initial_dist)
        print("Transition function")
        print(self.transition)
        print("Actions")
        print(self.actions)
        print("Reward function")
        print(self.rewards)

    def validate_transition(self):
        for state in self.transition.keys():
            for x in self.transition[state].values():
                validate_prob_axiom(x)

    def initialize_world(self, states, transition, rewards, initial_dist, actions, *args, **kwargs):

        self.states.update(states)

        self.actions.update(actions)
        self.transition.update(transition)

        self.__calculate_absorbing()
        self.rewards.update(rewards)
        self.initial_dist.update(initial_dist)

        validate_prob_axiom(self.initial_dist)
        self.validate_transition()

        if not isinstance(self, POMDP):  # Because for POMDP it calls POMDP.reset instead of MDP.reset
            self.reset()

    def reset(self):
        self.current_state = get_random(self.initial_dist)

    def take_action(self, action):
        if self.reached_absorbing():
            raise Exception("Taking action after reaching absorbing state")

        ret = self.rewards[self.current_state, action]
        self.current_state = get_random(self.transition[self.current_state][action])
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

    def display(self):
        super(POMDP, self).display()
        print("Observations")
        print(self.observations)
        print("Observation function")
        print(self.observation_function)

    def take_action(self, action):
        reward = super(POMDP, self).take_action(action)
        if not self.reached_absorbing():
            self.current_observation = get_random(self.observation_function[self.current_state])
        else:
            self.current_observation = None
        return reward

    def get_current_observation(self):
        return self.current_observation

    def reset(self):
        super(POMDP, self).reset()
        self.current_observation = get_random(self.observation_function[self.current_state])
