from experiment import Experiment
from parser import MDPParser, POMDPParser
from world import MDP, POMDP
from agent import Agent, AgentOnObservation


def get_world(parser):
    pomdp = POMDP(True)
    states = parser.parse_states()
    actions = parser.parse_actions()
    transition = parser.parse_transition_function()
    rewards = parser.parse_reward_function()
    initial_dist = parser.parse_init_dist()
    observations = parser.parse_observation()
    observation_function = parser.parse_observation_function()
    pomdp.initialize_world(states, transition, rewards, initial_dist, actions, observations, observation_function)
    return pomdp


def get_agent(parser, path):
    policy = parser.parse_policy(path)
    return AgentOnObservation(policy)


if __name__ == '__main__':
    parser = POMDPParser("./input/POMDP")

    pomdp = get_world(parser)
    algo = get_agent(parser, "pi.csv")
    simulation = Experiment(pomdp)
    num_episodes = 1
    # 1.
    print(simulation.estimate_return(algo, 1,100))
    # 2.
    # discounts = [0.25, 0.5, 0.75, 0.99]
    #
    # for discount in discounts:
    #     print(f"Return for {discount} : {simulation.estimate_return(algo, discount,num_episodes)}")
