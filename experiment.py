from functools import reduce

import matplotlib.pyplot as plt
import numpy as np

from operator import add
from agent import AgentOnObservation
from world import POMDP


class Simulator:
    def __init__(self, world, plot):
        self.world = world
        self.plot = plot

    def run(self, agent, discount=1, step=1, num_steps=1e2):
        self.world.reset()
        epi_return = 0
        current_discount = discount
        history = []
        t = 0
        step_reward = []
        while not self.world.reached_absorbing() and t < num_steps:
            current_state = self.world.get_current_state()

            if isinstance(agent, AgentOnObservation):
                current_observation = self.world.get_current_observation()
                next_action = agent.get_action(current_observation)
            else:
                next_action = agent.get_action(current_state)

            reward = self.world.take_action(next_action)
            if t != 0:
                epi_return += current_discount * reward

                if t < step:
                    step_reward.append(step_reward[-1] + current_discount * reward)

                current_discount *= discount
            else:
                epi_return += reward
                step_reward.append(reward)

            if isinstance(self.world, POMDP):
                current_observation = self.world.get_current_observation()
                history.append([current_state, current_observation, next_action, str(reward)])
            else:
                history.append([current_state, next_action, str(reward)])

            t += 1

        current_state = self.world.get_current_state()
        if isinstance(self.world, POMDP):
            current_observation = self.world.get_current_observation()
            history.append([current_state, current_observation])
        else:
            history.append([current_state])

        if t < step:
            step_reward = None
        else:
            step_reward = np.asarray(step_reward)

        return history, epi_return, step_reward


class Experiment(Simulator):

    def __init__(self, *args, **kwargs):
        super(Experiment, self).__init__(*args, **kwargs)

    def estimate_avg_return(self, agent, discount, num_episode, step=1):
        estimated_return = 0
        step_reward = np.zeros(step, dtype=float)
        points = []
        num_of_episode_to_sub = 0
        tot_epi_len = 0
        for i in range(1, num_episode + 1):
            epi_h, e_return, s_reward = self.run(agent, discount, step)
            estimated_return += e_return
            tot_epi_len += ''.join(reduce(add, epi_h)).count('s')  # Assumed all state names start with 's'
            if s_reward is None:
                num_of_episode_to_sub += 1
            else:
                step_reward = step_reward + s_reward
            if self.plot:
                points.append((i, estimated_return / i))

        if self.plot:
            plt.xlabel("Number of episodes")
            plt.ylabel("Estimated Return")
            plt.plot(*list(zip(*points)))
            plt.show()

        return step_reward / (
                num_episode - num_of_episode_to_sub), estimated_return / num_episode, tot_epi_len / num_episode
