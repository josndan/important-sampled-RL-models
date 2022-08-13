from matplotlib import pyplot as plt

from data_collector import DataCollector
from experiment import Experiment
from parser import MDPParser, POMDPParser
from utils import relative_error
from world import MDP, POMDP
from agent import Agent, AgentOnObservation
import time


def get_world(parser):
    pomdp = POMDP(False)
    states = parser.parse_states()
    actions = parser.parse_actions()
    transition = parser.parse_transition_function()
    rewards = parser.parse_reward_function()
    initial_dist = parser.parse_init_dist()
    observations = parser.parse_observation()
    observation_function = parser.parse_observation_function()
    pomdp.initialize_world(states, transition, rewards, initial_dist, actions, observations, observation_function)
    return pomdp


def get_agent(parser, path, policy_on_state=False):
    actions = parser.parse_actions()
    policy = parser.parse_policy(path)

    if policy_on_state:
        states = parser.parse_states()
        agent = Agent(states, actions)
    else:
        observations = parser.parse_observation()
        agent = AgentOnObservation(observations, actions)

    agent.initialize_policy(policy)
    return agent


def get_correcting_agent(data_collector, pi_agent, parser):
    states = parser.parse_states()
    actions = parser.parse_actions()
    mu_agent = Agent(states, actions)

    mu_policy = data_collector.get_correction_policy(pi_agent.policy)
    mu_agent.initialize_policy(mu_policy)

    return mu_agent


def timeit(func):
    """
    Decorator for measuring function's running time.
    """

    def measure_time(*args, **kw):
        start_time = time.time()
        result = func(*args, **kw)
        print("Processing time of %s(): %.2f seconds."
              % (func.__qualname__, time.time() - start_time))
        return result

    return measure_time


@timeit
def main(num_episodes):
    parser = POMDPParser("./input/POMDP")
    discount = 0

    pomdp = get_world(parser)
    print("Policy Pi")
    pi_agent = get_agent(parser, "pi.csv")
    print("Policy Mu_d")
    data_collecting_policy = get_agent(parser, "mu.csv", True)

    data_collector = DataCollector(pomdp, data_collecting_policy, num_epi=num_episodes)
    data_collector.collect()

    print("Corrected Policy")
    mu_agent = get_correcting_agent(data_collector, pi_agent, parser)

    simulation = Experiment(pomdp, plot=False)

    step = 1
    # number of single rewards do you want? step = 1 means just give me the ability to extract just the
    # first reward; step = 2 means give me the ability to extract the first as well as the second reward
    pi_step_reward, pi_return, pi_avg_len = simulation.estimate_avg_return(pi_agent, discount, num_episodes, step)
    mu_step_reward, mu_return, mu_avg_len = simulation.estimate_avg_return(mu_agent, discount, num_episodes, step)

    print(f"\npi {step} reward: {pi_step_reward[step - 1]}")
    print(f"\nmu {step} reward: {mu_step_reward[step - 1]}")
    print(f"\nAbsolute Error: {abs(pi_step_reward[step - 1] - mu_step_reward[step - 1]):0.5e}\n")

    print(f"\npi Return: {pi_return}")
    print(f"\npi avg len: {pi_avg_len}")
    print(f"\nmu Return: {mu_return}")
    print(f"\nmu avg len: {mu_avg_len}")
    print(f"\nAbsolute error: {abs(pi_return - mu_return)}\n")



if __name__ == '__main__':
    points = []
    num_epi = [1e4]

    for i, n in enumerate(num_epi):
        print(f"\n\nIn simulation {i + 1}")
        error = main(int(n))
        points.append((n, error))

    # plt.xlabel("Number of episodes")
    # plt.ylabel("Relative error in pi vs mu")
    # plt.plot(*list(zip(*points)))
    # plt.show()
