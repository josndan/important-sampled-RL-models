import operator
from functools import reduce
from collections import Counter
import numpy as np
from tqdm import tqdm
from agent import AgentOnObservation
from world import POMDP
from joblib import Parallel, delayed


def run(agent, world, discount, step, epi_len, return_history=False):
    world.reset()
    epi_return = 0
    current_discount = discount
    if return_history:
        history = []
    t = 0
    step_reward = []
    observation_visitation = Counter()
    state_visitation = Counter()

    while not world.reached_absorbing() and t < epi_len:
        current_state = world.get_current_state()
        state_visitation[current_state] += 1
        if isinstance(world, POMDP):
            current_observation = world.get_current_observation()
            observation_visitation[current_observation] += 1
        if isinstance(agent, AgentOnObservation):
            if current_observation is None:
                raise Exception("Current Observation is None")  # sanity check this should never happen
            next_action = agent.get_action(current_observation)
        else:
            next_action = agent.get_action(current_state)

        reward = world.take_action(next_action)
        if t != 0:
            epi_return += current_discount * reward

            # if t < step: # This would be step return
            #     step_reward.append(step_reward[-1] + current_discount * reward)

            current_discount *= discount
        else:
            epi_return += reward
            # step_reward.append(reward)

        if t < step:
            step_reward.append(reward)

        if return_history:
            if isinstance(world, POMDP):
                history.append([current_state, current_observation, next_action,
                                'r' + str(reward)])  # Precautionary Assumption that 'r' is not a state or
                # observation name
            else:
                history.append([current_state, next_action, 'r' + str(reward)])

        t += 1

    if return_history:
        current_state = world.get_current_state()
        if isinstance(world, POMDP):
            current_observation = world.get_current_observation()
            history.append([current_state, current_observation])
        else:
            history.append([current_state])

    if t < step:
        raise Exception("Time step less than episode length")

    if return_history:
        return history, epi_return, np.asarray(step_reward)
    else:
        return epi_return, np.asarray(step_reward), observation_visitation, state_visitation


class Experiment:

    def __init__(self, world_factory):
        self.world_factory = world_factory

    def estimate_avg_return(self, agent, discount, num_episode, epi_len, step=None, parallel=True):
        if step is None:
            step = epi_len

        if step > epi_len:
            raise Exception("time steps length greater than episode length")

        step = int(step)

        estimated_return = 0
        all_returns = []
        step_reward = np.zeros(step)
        observation_visitations = Counter()
        state_visitations = Counter()

        if parallel:
            for epi_ret, step_reward, observation_visitation, state_visitation in tqdm(Parallel(n_jobs=4)(
                    delayed(run)(agent, self.world_factory(), discount, step, epi_len, False) for _ in
                    range(num_episode))):
                estimated_return += epi_ret
                all_returns.append(epi_ret)
                step_reward += step_reward
                observation_visitations += observation_visitation
                state_visitations += state_visitation

            # estimated_return, step_reward = reduce(lambda a, b: tuple(map(operator.add, a, b)), result,
            #                                        (0, np.zeros(step)))
        else:
            for i in tqdm(range(1, num_episode + 1)):
                e_return, s_reward, observation_visitation, state_visitation = run(agent, self.world_factory(),
                                                                                   discount, step, epi_len)
                estimated_return += e_return
                all_returns.append(e_return)
                observation_visitations += observation_visitation
                state_visitations += state_visitation

                step_reward = step_reward + s_reward

        normalization_factor_obs = sum(observation_visitations.values())

        normalization_factor_states = sum(state_visitations.values())

        for key in observation_visitations:
            observation_visitations[key] /= normalization_factor_obs

        for key in state_visitations:
            state_visitations[key] /= normalization_factor_states

        return step_reward / num_episode, estimated_return / num_episode, observation_visitations, state_visitations, all_returns
