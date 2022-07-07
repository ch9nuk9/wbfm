from numba import jit


@jit(nopython=True)
def find_first(item, vec: list):
    """
    return the index of the first occurrence of item in vec

    from: https://stackoverflow.com/questions/7632963/numpy-find-first-index-of-value-fast
    """
    for i, val in enumerate(vec):
        if item == val:
            return i
    return -1
