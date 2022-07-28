from experiment import Experiment
from parser import MDPParser
from world import World
from agent import Agent


def get_world(parser):
    mdp = World()
    states = parser.parse_states()
    transition = parser.parse_transition_function()
    rewards = parser.parse_reward_function()
    initial_dist = parser.parse_init_dist()
    mdp.initialize_world(states, transition, rewards, initial_dist)
    return mdp


def get_agent(parser):
    policy = parser.parse_policy("policy.csv")
    return Agent(policy)


if __name__ == '__main__':
    parser = MDPParser("./input/MDP")

    mdp = get_world(parser)
    algo = get_agent(parser)
    simulation = Experiment(mdp)

    num_episodes = 150000
    # 1.
    # print(simulation.estimate_return(algo, 1, num_episodes))

    # 2.
    # discounts = [0.25, 0.5, 0.75, 0.99]
    #
    # for discount in discounts:
    #     print(f"Return for {discount} : {simulation.estimate_return(algo, discount,num_episodes)}")
