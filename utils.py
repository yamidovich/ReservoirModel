from math import ceil, floor


def check_if_numerical(x):
    if (type(x) == float) | (type(x) == int):
        return True
    return False


def check_half(x) -> bool:
    """
    check if x is a kind of int + 0.5
    :param x: float value
    :return: true if x is of expected kind
    """
    if ceil(x) == x + 0.5:
        return True
    return False


def check_int(x) -> bool:
    """
    check if x is int
    :param x: float or int
    :return: true if x is int value
    """
    if int(x) == x:
        return True
    return False


def two_dim_index_to_one(i: int, j: int, ny: int) -> int:
    return ny * i + j


def one_d_index_to_two(one_d: int, ny: int):
    i = int(one_d / ny)
    j = one_d % ny
    return i, j
