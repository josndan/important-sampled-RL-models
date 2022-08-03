import matplotlib.pyplot as plt

from agent import AgentOnObservation
from world import POMDP


class Simulator:
    def __init__(self, world, plot):
        self.world = world
        self.plot = plot

    def run(self, agent, discount=1):
        self.world.reset()
        epi_return = 0
        current_discount = discount
        history = []
        t = 0
        first_reward = 0
        while not self.world.reached_absorbing():
            current_state = self.world.get_current_state()

            if isinstance(agent, AgentOnObservation):
                current_observation = self.world.get_current_observation()
                next_action = agent.get_action(current_observation)
            else:
                next_action = agent.get_action(current_state)

            reward = self.world.take_action(next_action)
            if t != 0:
                epi_return += current_discount * reward
                current_discount *= discount
            else:
                epi_return += reward
                first_reward = reward

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
        return history, epi_return, first_reward


class Experiment(Simulator):

    def __init__(self, *args, **kwargs):
        super(Experiment, self).__init__(*args, **kwargs)

    def estimate_avg_return(self, agent, discount, num_episode):
        estimated_return = 0
        first_reward = 0
        points = []
        for i in range(1, num_episode + 1):
            _, e_return, f_reward = self.run(agent, discount)
            estimated_return += e_return
            first_reward += f_reward
            if self.plot:
                points.append((i, estimated_return / i))

        if self.plot:
            plt.xlabel("Number of episodes")
            plt.ylabel("Estimated Return")
            plt.plot(*list(zip(*points)))
            plt.show()

        return first_reward/num_episode, estimated_return / num_episode
