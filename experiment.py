import matplotlib.pyplot as plt

from agent import AgentOnObservation


class Experiment:

    def __init__(self, world,plot):
        self.world = world
        self.plot = plot

    def run_episode(self, agent, discount):
        self.world.reset()
        epi_return = 0
        current_discount = discount
        if isinstance(agent, AgentOnObservation):
            while not self.world.reached_absorbing():
                current_observation = self.world.get_current_observation()
                next_action = agent.get_action(current_observation)
                reward = self.world.take_action(next_action)
                epi_return += discount * reward
                current_discount *= discount
        else:
            while not self.world.reached_absorbing():
                current_state = self.world.get_current_state()
                next_action = agent.get_action(current_state)
                reward = self.world.take_action(next_action)
                epi_return += discount * reward
                current_discount *= discount

        return epi_return

    def estimate_return(self, agent, discount, num_episode):
        estimated_return = 0
        points = []
        for i in range(1, num_episode + 1):
            estimated_return += self.run_episode(agent, discount)
            points.append((i, estimated_return / i))

        if self.plot:
            plt.xlabel("Number of episodes")
            plt.ylabel("Estimated Return")
            plt.plot(*list(zip(*points)))
            plt.show()

        return estimated_return / num_episode
