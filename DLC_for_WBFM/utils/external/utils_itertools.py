import random


def random_combination(iterable, r):
    """Random selection (single samples) from itertools.combinations(iterable, r)"""
    pool = tuple(iterable)
    n = len(pool)
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


def reverse_dict(d):
    return {v: k for k, v in d.items()}
