import random


def get_random(dist):
    return random.choices(*list(zip(*dist.items())))[0]
