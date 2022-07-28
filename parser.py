import pandas as pd
from pathlib import PurePath
from collections import defaultdict


class MDPParser:
    def __init__(self, model_path):
        self.model_path = PurePath(model_path)

    def parse_states(self):
        states_df = pd.read_csv(self.model_path.joinpath("states.csv").as_posix(), skipinitialspace=True)
        return set(states_df["states"].values)

    def parse_transition_function(self):
        transition_function = pd.read_csv(self.model_path.joinpath("transition_function.csv").as_posix(),
                                          skipinitialspace=True)
        result = defaultdictWhichPrints(lambda: defaultdictWhichPrints(lambda: defaultdictWhichPrints(int)))
        for index, row in transition_function.iterrows():
            f, a, t, p = row[0], row[1], row[2], row[3]
            result[f][a][t] = p
        return result

    def parse_reward_function(self):
        reward_function = pd.read_csv(self.model_path.joinpath("reward_function.csv").as_posix(), skipinitialspace=True)
        result = defaultdictWhichPrints(int)
        for index, row in reward_function.iterrows():
            s, a, p = row[0], row[1], row[2]
            result[s, a] = p
        return result

    def parse_policy(self, policyPath):
        policy = pd.read_csv(self.model_path.joinpath(policyPath).as_posix(), skipinitialspace=True)
        result = defaultdictWhichPrints(lambda: defaultdictWhichPrints(int))
        for index, row in policy.iterrows():
            s, a, p = row[0], row[1], row[2]
            result[s][a] = p
        return result

    def parse_init_dist(self):
        init_dist = pd.read_csv(self.model_path.joinpath("initial_dist.csv").as_posix(), skipinitialspace=True)
        result = defaultdictWhichPrints(int)
        for index, row in init_dist.iterrows():
            s, p = row[0], row[1]
            result[s] = p
        return result


class defaultdictWhichPrints(defaultdict):
    def __init__(self, *args, **kargs):
        super(defaultdictWhichPrints, self).__init__(*args, **kargs)

    def __repr__(self):
        return dict.__repr__(self)
