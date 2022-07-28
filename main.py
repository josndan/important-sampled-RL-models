from experiment import Experiment
from world import World
from agent import Agent


def get_world():
    mdp = World()
    states = {"s1", "s2", "s3", "s4", "s5", "s6"}
    transition = {"s1": {"a1": {"s2": 0.6, "s5": 0.4}, "a2": {"s2": 0.1, "s5": 0.9}}
        , "s2": {"a1": {"s3": 0.2, "s4": 0.8}, "a2": {"s3": 0.5, "s4": 0.5}}
        , "s5": {"a1": {"s4": 1}, "a2": {"s4": 1}}
        , "s6": {"a1": {"s5": 1}, "a2": {"s5": 1}}}
    rewards = {("s1", "a1"): 7
        , ("s1", "a2"): 4
        , ("s2", "a1"): 9
        , ("s2", "a2"): -4
        , ("s5", "a1"): -2
        , ("s5", "a2"): 6
        , ("s6", "a1"): 0
        , ("s6", "a2"): 1}
    initial_dist = {"s1": 0.3, "s6": 0.7}
    mdp.initialize_world(states, transition, rewards, initial_dist)
    return mdp


def get_agent():
    policy = {"s1": {"a1": 0.3, "a2": 0.7}
        , "s2": {"a1": 0.1, "a2": 0.9}
        , "s3": {"a1": 1}
        , "s4": {"a1": 1}
        , "s5": {"a1": 0.25, "a2": 0.75}
        , "s6": {"a1": 0.4, "a2": 0.6}}
    return Agent(policy)


if __name__ == '__main__':
    mdp = get_world()
    algo = get_agent()
    simulation = Experiment(mdp)

    num_episodes = 150000

    # 1.
    print(simulation.estimate_return(algo, 1, num_episodes))

    # 2.
    # discounts = [0.25, 0.5, 0.75, 0.99]
    #
    # for discount in discounts:
    #     print(f"Return for {discount} : {simulation.estimate_return(algo, discount,num_episodes)}")
