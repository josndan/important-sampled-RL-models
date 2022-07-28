import random
import copy


def get_random(dist):
    return random.choices(*list(zip(*dist.items())))[0]


class CustomDefaultDict(dict):
    def __init__(self, set_to_check, default):
        super(CustomDefaultDict, self).__init__()
        self.set_to_check = set_to_check
        self.default = default

    def __getitem__(self, key):
        if key in self.set_to_check and key not in self:
            return copy.deepcopy(self.default)
        return dict.__getitem__(self, key)

    def update(self, other, **kwargs):
        for key in other:
            if isinstance(self[key],CustomDefaultDict):
                temp = self[key]
                temp.update(other[key])
                self[key] = temp
            else:
                self[key] = other[key]
