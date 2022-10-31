import random
import numpy as np
import copy
from math import isclose
import multiprocessing


def get_random(dist):
    return random.choices(*list(zip(*dist.items())))[0]


# def get_random(dist):
#     key, p = list(zip(*dist.items()))
#     return np.random.default_rng().choice(key, p=p)


def parallelize(func, args):
    with multiprocessing.Pool(multiprocessing.cpu_count()) as p:
        return p.starmap(func, args)


def normalize(dist):
    dist = dist.copy()
    tot = sum(dist.values())
    for qtn in dist:
        dist[qtn] /= tot

    return dist


def validate_prob_axiom(dist):
    if not isclose(sum(dist.values()), 1, rel_tol=1e-1):
        # raise Exception("Probabilities must add up to one")
        print("Probabilities don't add up")


def relative_error(x, y):
    return max(abs(x - y) / (x if x != 0 else float("-inf")), abs(x - y) / (y if y != 0 else float("-inf")))

#Nested Defaultdict is better name
class CustomDefaultDict(dict):
    def __init__(self, set_to_check, default):
        super(CustomDefaultDict, self).__init__()
        self.set_to_check = set_to_check
        self.default = default

    def update_set_to_check(self, set_to_check):
        self.set_to_check = set_to_check

    def __getitem__(self, key):
        if key in self.set_to_check and key not in self:
            self[key] = copy.deepcopy(self.default)
            return self[key]
        return dict.__getitem__(self, key)

    def update(self, other, **kwargs):
        for key in other:
            if isinstance(self[key], CustomDefaultDict):
                temp = self[key]
                temp.update(other[key])
                self[key] = temp
            else:
                self[key] = other[key]
