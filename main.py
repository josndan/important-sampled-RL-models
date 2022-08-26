from collections import Counter

import matplotlib
import numpy as np
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
def main(num_episodes, verbose=True):
    parser = POMDPParser("./input/POMDP")
    discount = 0

    pomdp = get_world(parser)

    pi_agent = get_agent(parser, "pi.csv")
    data_collecting_policy = get_agent(parser, "mu.csv", True)

    data_collector = DataCollector(pomdp, data_collecting_policy, num_epi=num_episodes, epi_len=10)
    data_collector.collect()
    #
    # print("History")
    # print(data_collector.history)
    #
    mu_agent = get_correcting_agent(data_collector, pi_agent, parser)

    if verbose:
        print("Policy Pi")
        print(pi_agent.policy)

        print("Policy Mu_d")
        print(data_collecting_policy.policy)

        print("Corrected Policy")
        print(mu_agent.policy)

    simulation = Experiment(pomdp, plot=False)

    step = 100
    # number of single rewards do you want? step = 1 means just give me the ability to extract just the
    # first reward; step = 2 means give me the ability to extract the first as well as the second reward
    pi_step_reward, pi_return, pi_avg_len = simulation.estimate_avg_return(pi_agent, discount,
                                                                           num_episodes, step)
    mu_step_reward, mu_return, mu_avg_len = simulation.estimate_avg_return(mu_agent, discount, num_episodes, step)

    # if verbose:
    #     for step_len in range(int(step)):
    #         print(f"\npi {step_len + 1} reward: {pi_step_reward[step_len]}")
    #         print(f"\nmu {step_len + 1} reward: {mu_step_reward[step_len]}")
    #         print(f"\nAbsolute Error: {abs(pi_step_reward[step_len] - mu_step_reward[step_len]):0.5e}\n")

    print(f"\npi Return: {pi_return}")
    print(f"\nmu Return: {mu_return}")
    print(f"\nAbsolute error: {abs(pi_return - mu_return):0.5e}\n")
    print()
    print(f"\npi avg len: {pi_avg_len}")
    print(f"\nmu avg len: {mu_avg_len}")

    return pi_step_reward, mu_step_reward, pi_return, mu_return


if __name__ == '__main__':
    points = []
    num_epi = [1e5]

    fig = plt.figure()
    ax = fig.add_subplot()
    plt.xlabel("Step")
    plt.ylabel("Absolute Error")

    for i, n in enumerate(num_epi):
        print(f"\n\nIn simulation {i + 1}")
        pi_step_reward, mu_step_reward, _, _ = main(int(n))
        # points.append((n, error))

        y = np.absolute(pi_step_reward - mu_step_reward)
        x = np.arange(len(y))
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        ax.plot(y, label=f"Number of trials {int(n):0.1e}")
        ax.plot(x, p(x))

    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2e'))
    plt.legend()
    plt.show()
    # plt.xlabel("Number of episodes")
    # plt.ylabel("Relative error in pi vs mu")
    # plt.plot(*list(zip(*points)))
    # plt.show()
