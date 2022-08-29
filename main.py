import sys
from collections import Counter
from typing import Callable

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from data_collector import DataCollector
from experiment import Experiment
from parser import POMDPParser
from utils import get_random
from world import POMDP
from agent import Agent, AgentOnObservation
import time


def get_world_factory(parser):
    states = parser.parse_states()
    actions = parser.parse_actions()
    transition = parser.parse_transition_function()
    rewards = parser.parse_reward_function()
    initial_dist = parser.parse_init_dist()
    observations = parser.parse_observation()
    observation_function = parser.parse_observation_function()

    def factory():
        pomdp = POMDP()
        pomdp.initialize_world(states, transition, rewards, initial_dist, actions, observations, observation_function)
        return pomdp

    return factory


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


def get_baseline_estimation(num_episode):
    dist = {"t1": 0.4, "t2": 0.6}
    estimate = Counter()
    for i in range(num_episode):
        estimate[get_random(dist)] += 1
    for i in estimate:
        print(f"{abs(estimate[i] / num_episode - dist[i]):0.5e}")


def get_baseline_equal_policy(num_episode, epi_len=10):
    parser = POMDPParser("./input/POMDP")
    pomdp_factory = get_world_factory(parser)
    policy = get_agent(parser, "mu.csv", True)

    simulation = Experiment(pomdp_factory)

    s1_step_reward, _ = simulation.estimate_avg_return(policy, 0,
                                                       num_episode, epi_len)
    s2_step_reward, _ = simulation.estimate_avg_return(policy, 0, num_episode, epi_len)

    print("Step reward stats")
    error = np.abs(s1_step_reward - s2_step_reward)
    print(f"\nmin error: {np.min(error):0.5e}")
    print(f"\nmax error: {np.max(error):0.5e}")
    print(f"\naverage: {np.average(error):0.5e}\n")


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
def simulate(num_episodes, verbose=True, epi_len=10):
    parser = POMDPParser("./input/POMDP")
    discount = 0.8

    pomdp_factory = get_world_factory(parser)

    pi_agent = get_agent(parser, "pi.csv")
    data_collecting_policy = get_agent(parser, "mu.csv", True)

    data_collector = DataCollector(pomdp_factory(), data_collecting_policy, num_epi=int(1e4), epi_len=10)
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
        print()
        sys.stdout.flush()

    simulation = Experiment(pomdp_factory)

    # number of single rewards do you want? step = 1 means just give me the ability to extract just the
    # first reward; step = 2 means give me the ability to extract the first as well as the second reward
    pi_step_reward, pi_return = simulation.estimate_avg_return(pi_agent, discount,
                                                               num_episodes, epi_len)
    mu_step_reward, mu_return = simulation.estimate_avg_return(mu_agent, discount, num_episodes, epi_len)

    # if verbose:
    #     for step_len in range(int(step)):
    #         print(f"\npi {step_len + 1} reward: {pi_step_reward[step_len]}")
    #         print(f"\nmu {step_len + 1} reward: {mu_step_reward[step_len]}")
    #         print(f"\nAbsolute Error: {abs(pi_step_reward[step_len] - mu_step_reward[step_len]):0.5e}\n")

    print(f"\npi Return: {pi_return}")
    print(f"\nmu Return: {mu_return}")
    print(f"\nAbsolute error: {abs(pi_return - mu_return):0.5e}\n")
    print("Step reward stats")
    error = np.abs(pi_step_reward - mu_step_reward)
    print(f"\nmin error: {np.min(error):0.5e}")
    print(f"\nmax error: {np.max(error):0.5e}")
    print(f"\naverage: {np.average(error):0.5e}\n")

    return pi_step_reward, mu_step_reward, pi_return, mu_return


def main(num_epi):
    # get_baseline_equal_policy(int(1e5))

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # plt.xlabel("Step")
    # plt.ylabel("Absolute Error")

    for i, n in enumerate(num_epi):
        print(f"\n\nIn simulation {i + 1}")
        pi_step_reward, mu_step_reward, _, _ = simulate(int(n))
        # y = np.absolute(pi_step_reward - mu_step_reward)
        # x = np.arange(1, len(y) + 1)
        # z = np.polyfit(x, y, 1)
        # p = np.poly1d(z)
        # ax.scatter(x, y, label=f"Number of trials {int(n):0.1e}")
        # ax.plot(x, p(x))

    # ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2e'))
    # plt.legend()
    # plt.show()


if __name__ == '__main__':
    num_epi = [1e5]
    main(num_epi)
    # get_baseline_equal_policy(int(num_epi[0]))
