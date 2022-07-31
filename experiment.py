import matplotlib.pyplot as plt

from agent import AgentOnObservation


class Simulator:
    def __init__(self, world, plot):
        self.world = world
        self.plot = plot

    def run(self, agent, discount=1):
        self.world.reset()
        epi_return = 0
        current_discount = discount
        history = []
        if isinstance(agent, AgentOnObservation):
            while not self.world.reached_absorbing():
                current_observation = self.world.get_current_observation()
                current_state = self.world.get_current_state()
                next_action = agent.get_action(current_observation)
                reward = self.world.take_action(next_action)
                epi_return += discount * reward
                current_discount *= discount
                history.append((current_state, current_observation, next_action, reward))
        else:
            while not self.world.reached_absorbing():
                current_state = self.world.get_current_state()
                next_action = agent.get_action(current_state)
                reward = self.world.take_action(next_action)
                epi_return += discount * reward
                current_discount *= discount
                history.append((current_state, next_action, reward))

        return history, epi_return


class Experiment(Simulator):

    def __init__(self, *args, **kwargs):
        super(Experiment, self).__init__(*args, **kwargs)

    def estimate_return(self, agent, discount, num_episode):
        estimated_return = 0
        points = []
        for i in range(1, num_episode + 1):
            estimated_return += self.run(agent, discount)[1]
            points.append((i, estimated_return / i))

        if self.plot:
            plt.xlabel("Number of episodes")
            plt.ylabel("Estimated Return")
            plt.plot(*list(zip(*points)))
            plt.show()

        return estimated_return / num_episode
