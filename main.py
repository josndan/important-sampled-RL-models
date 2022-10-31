import sys
from collections import Counter
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
import seaborn as sns


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
        agent = Agent(observations, actions)

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


# Measure epistemic uncertainty due to finite number of trails
def get_baseline_equal_policy(num_episode, epi_len=10, discount=1):
    parser = POMDPParser("./input/POMDP")
    pomdp_factory = get_world_factory(parser)
    policy = get_agent(parser, "data_collecting_states.csv", True)

    simulation = Experiment(pomdp_factory)

    avg_step_reward_1, avg_ret_1, _, _, all_returns = simulation.estimate_avg_return(policy, discount,
                                                                                  num_episode, epi_len)
    avg_step_reward_2, avg_ret_2, _, _, all_returns_2 = simulation.estimate_avg_return(policy, discount, num_episode,
                                                                                    epi_len)

    print("Baseline")
    print("\nDifference in average return stats")
    print("Variance ", abs(avg_ret_2 - avg_ret_1))

    error = np.abs(avg_step_reward_1 - avg_step_reward_2)
    print("Difference in average step reward stats")
    print(f"min variance: {np.min(error):0.5e}")
    print(f"max variance: {np.max(error):0.5e}")
    print(f"average variance: {np.average(error):0.5e}\n")


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
def simulate(num_episodes, verbose=True, epi_len=10, discount=1):
    parser = POMDPParser("./input/POMDP")

    pomdp_factory = get_world_factory(parser)

    pi_agent = get_agent(parser, "pi.csv")
    data_collecting_policy = get_agent(parser, "data_collecting_states.csv", True)

    # data_collecting_policy = get_agent(parser, "data_collecting_obs.csv")

    data_collector = DataCollector(pomdp_factory(), data_collecting_policy, num_epi=int(1e4), epi_len=10)
    data_collector.collect()
    #
    # print("History")
    # print(data_collector.history)
    #

    m_hat_factory = data_collector.get_estimated_model()

    mu_agent = get_correcting_agent(data_collector, pi_agent, parser)

    print("\nThe Policy are\n")
    if verbose:
        print("Data collecting policy")
        print(data_collecting_policy.policy)

        print("Policy Pi")
        print(pi_agent.policy)

        print("Policy Mu")
        print(mu_agent.policy)
        print()
        sys.stdout.flush()

    simulation = Experiment(pomdp_factory)

    simulation_estimate = Experiment(m_hat_factory)

    # number of single rewards do you want? step = 1 means just give me the ability to extract just the
    # first reward; step = 2 means give me the ability to extract the first as well as the second reward
    pi_step_reward, pi_return, pi_observation_visitations, pi_state_visitations, all_returns_pi = simulation_estimate.estimate_avg_return(pi_agent,
                                                                                                          discount,
                                                                                                          num_episodes,
                                                                                                          epi_len)
    mu_step_reward, mu_return, mu_observation_visitations, mu_state_visitations, all_returns_mu = simulation.estimate_avg_return(mu_agent,
                                                                                                          discount,
                                                                                                          num_episodes,
                                                                                                          epi_len)

    # if verbose:
    #     for step_len in range(int(step)):
    #         print(f"\npi {step_len + 1} reward: {pi_step_reward[step_len]}")
    #         print(f"\nmu {step_len + 1} reward: {mu_step_reward[step_len]}")
    #         print(f"\nAbsolute Error: {abs(pi_step_reward[step_len] - mu_step_reward[step_len]):0.5e}\n")

    print(f"\npi Average Return: {pi_return}")
    print(f"mu Average Return: {mu_return}")
    print(f"Absolute error in average return: {abs(pi_return - mu_return):0.5e}\n")

    # plt.hist(all_returns_1)
    # plt.hist(all_returns_2)
    # plt.show()

    print("Step reward stats")
    error = np.abs(pi_step_reward - mu_step_reward)
    print(f"min error in difference in average rewards: {np.min(error):0.5e}")
    print(f"max error in difference in average rewards: {np.max(error):0.5e}")
    print(f"average error in difference in average rewards: {np.average(error):0.5e}\n")
    print()
    print("Pi state visitation: (which is actually observations)")
    print(pi_state_visitations)
    print()
    print("Mu Observation visitation:")
    print(mu_observation_visitations)

    return pi_step_reward, mu_step_reward, pi_return, mu_return,all_returns_pi,all_returns_mu


def main(num_trials_in_each_simulation, epi_len=10):
    # get_baseline_equal_policy(int(1e5))

    fig, (ax1, ax2,ax3) = plt.subplots(3)
    discount = 0.9
    for i, n in enumerate(num_trials_in_each_simulation):
        print(f"\n\nIn simulation {i + 1}")

        get_baseline_equal_policy(int(n), epi_len=epi_len,
                                  discount=discount)
        pi_step_reward, mu_step_reward, _, _,all_return_pi,all_return_mu = simulate(int(n), epi_len=epi_len,
                                                        discount=discount)

        y = np.absolute(pi_step_reward - mu_step_reward)

        x = np.arange(1, len(y) + 1)
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)

        ax3.plot(x, p(x))
        ax3.scatter(x, y, label=f"Number of trials {int(n):0.1e}")

        ax1.hist(all_return_pi)
        ax2.hist(all_return_mu)

    ax1.set_xlabel("Return of pi")
    ax1.set_ylabel("Frequency")

    ax2.set_xlabel("Return of mu")
    ax2.set_ylabel("Frequency")

    ax1.get_shared_x_axes().join(ax1,ax2)

    ax3.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2e'))
    ax3.legend()
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Absolute Error")

    ax3.title.set_text("Absolute error vs time step")

    fig.tight_layout()

    plt.show()


if __name__ == '__main__':
    sns.set()
    num_trials_in_each_simulation = [1e5] #Can't be more than 1 element because of histogram plot
    main(num_trials_in_each_simulation)
