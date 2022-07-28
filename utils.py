import random


def get_random(dist):
    return random.choices(*list(zip(*dist.items())))[0]


class CustomDefaultDict(dict):
    def __init__(self, set_to_check, default):
        super(CustomDefaultDict, self).__init__()
        self.set_to_check = set_to_check
        self.default = default

    def __getitem__(self, key):
        if key in self.set_to_check and key not in self:
            return self.default
        return dict.__getitem__(self, key)
