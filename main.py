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

    mu_policy = data_collector.get_correction_policy(pi_agent.policy, False)
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
def main():
    parser = POMDPParser("./input/POMDP")
    num_episodes = 100000
    discount = 1

    pomdp = get_world(parser)
    pi_agent = get_agent(parser, "pi.csv")
    data_collecting_policy = get_agent(parser, "mu.csv", True)

    data_collector = DataCollector(pomdp, data_collecting_policy, num_epi=num_episodes)
    data_collector.collect()

    mu_agent = get_correcting_agent(data_collector, pi_agent, parser)

    simulation = Experiment(pomdp, plot=False)

    pi_first_reward, pi_return = simulation.estimate_avg_return(pi_agent, discount, num_episodes)
    mu_first_reward, mu_return = simulation.estimate_avg_return(mu_agent, discount, num_episodes)

    print(f"\npi first reward: {pi_first_reward}")
    print(f"\nmu first reward: {mu_first_reward}")
    print(f"\nRelative Error: {relative_error(pi_first_reward, mu_first_reward)}\n")

    print(f"\npi Return: {pi_return}")
    print(f"\nmu Return: {mu_return}")
    print(f"\nRelative error: {relative_error(pi_return, mu_return)}\n")


if __name__ == '__main__':
    main()
