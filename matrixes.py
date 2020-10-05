from sklearn.datasets import make_moons, make_circles
import pandas as pd
import numpy as np
from math import floor, ceil


def get_saturation_scattered(n_samples=500, noise=0.4, random_state=None) -> tuple:
    """
    returns scatter points for interpritation as water and oil
    shape - moons
    :param n_samples: number of samples
    :param noise: how water and oil are mixtured
    :param random_state: rand state
    :return: (x, y). x - np.ndarray(n_samples, 2) for 2 coordinates. y - {0, 1} oil or water
    """
    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)


def get_grid_from_scattered(ds: pd.DataFrame,
                            y_min: float = -1.5, y_max: float = 1.5,
                            x_min: float = -1.5, x_max: float = 1.5,
                            n_x: int = 25, n_y: int = 25) -> np.ndarray:
    """
    from scattered df with point returns a greed as a matrix.
    :param ds: (np.ndarray(2, n), np.ndarray(n) in {1, 2}^n )
    :param y_min: as distribution for scattered
    bound may be generated randomly here is a handcrafted bound
    :param y_max: as distribution for scattered
    bound may be generated randomly here is a handcrafted bound
    :param x_min: as distribution for scattered
    bound may be generated randomly here is a handcrafted bound
    :param x_max: as distribution for scattered
    bound may be generated randomly here is a handcrafted bound
    :param n_x: number of cells in x component
    :param n_y: number of cells in x component
    :return: np.ndarray(n_y, n_xx) in [0,1]^{n_x * n_y} or some cells can be equal
     to -1 as cells with no samples
    """
    xs = np.linspace(x_min, x_max, n_x + 1)
    ys = np.linspace(y_min, y_max, n_y + 1)
    greed_matrix = (-1) * np.ones((n_x, n_y))

    for ix in range(len(xs) - 1):
        for iy in range(len(ys) - 1):
            x_filter = (ds[0][:, 0] > xs[ix]) & (ds[0][:, 0] < xs[ix + 1])
            y_filter = (ds[0][:, 1] > ys[iy]) & (ds[0][:, 1] < ys[iy + 1])
            _filter = (y_filter & x_filter)
            if _filter.sum() > 0:
                greed_matrix[ix][iy] = ds[1][_filter].mean()

    return greed_matrix


def get_depth_scattered(n_samples=500, noise=0.3, random_state=None, factor=0.4):
    ds = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    y = ds[1] * 1.5 - 0.5 * np.ones(ds[1].shape)
    return ds[0], y


def get_porosity_scattered(n_samples=500, noise=0.4, random_state=None, factor=0.4):
    ds = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    y = ds[1] * 1.5 - 0.5 * np.ones(ds[1].shape)
    return ds[0], y


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


class KMatrix:
    """
    class for matrix k. we got inter block values, but what do we have on boundaries...
    so we have to call [1.5, 4.5] to get proper inter block value
    """

    def __init__(self, k_values, dy_matrix, dx_matrix):
        self.k = k_values
        self.dy = dy_matrix
        self.dx = dx_matrix
        # what if case is simple and no matrix for k and dx, dy needed and won't be called
        # but if k is a matrix and grid dx is a constant step -
        if (type(self.k) == np.ndarray) & check_if_numerical(self.dx) & check_if_numerical(self.dy):
            self.dx = np.ones(self.k.shape) * self.dx
            self.dy = np.ones(self.k.shape) * self.dy
        self.scalar = False
        if check_if_numerical(self.k) & check_if_numerical(self.dx) & check_if_numerical(self.dy):
            self.scalar = True

    def __getitem__(self, item):
        """
        returns an inter block value or in-block...
        :param item: [1.5, 3.5], not [1.5][3.5]
        :return:
        """
        # simplest case
        if self.scalar:
            return self.k
        # major case - tuple
        if (type(item) == tuple) & (len(item) == 2):
            i, j = item

            if check_int(i) & check_half(j):
                out = self.dy[floor(j)] / self.k[floor(i), floor(j)]
                out += self.dy[ceil(j)] / self.k[floor(i), ceil(j)]
                out = 1 / out
                out *= self.dy[floor(j)] + self.dy[int(i)][ceil(j)]
                return out

            if check_half(i) & check_int(j):
                out = self.dx[floor(i)] / self.k[floor(i), floor(j)]
                out += self.dx[ceil(i)] / self.k[ceil(i), floor(j)]
                out = 1 / out
                out *= self.dx[floor(i)] + self.dx[ceil(i)]
                return out
            # if for some reason not boundary condition
            if check_int(i) & check_int(j):
                return self.k[item]

        elif type(item) == int:
            return self.k[item]


class DMatrix:
    """
    class for matrix of depth. we got inter block values, but what do we have on boundaries...
    so we have to call [1, 4.5] to get proper inter block value
    """

    def __init__(self, d_values):
        self.__d = d_values

    def __getitem__(self, item):
        """
        returns an inter block value or in-block...
        :param item: [1.5, 3.5], not [1.5][3.5]
        :return:
        """
        # simplest case
        if check_if_numerical(self.__d):
            return self.__d
        # major case - tuple
        if (type(item) == tuple) & (len(item) == 2):
            i, j = item

            if check_int(i) & check_half(j):
                out = self.__d[floor(i), floor(j)] + self.__d[floor(i), ceil(j)]
                out *= 0.5
                return out

            if check_half(i) & check_int(j):
                out = self.__d[floor(i), floor(j)] + self.__d[ceil(i), floor(j)]
                out *= 0.5
                return out
            # if for some reason not boundary condition
            if check_int(i) & check_int(j):
                return self.__d[item]

        elif type(item) == int:
            return self.__d[item]


class TInterBlockMatrix:
    """
    class for matrix of depth. we got inter block values, but what do we have on boundaries...
    so we have to call [1, 4.5] to get proper inter block value
    """

    def __init__(self,
                 k_values: np.ndarray,
                 dx_matrix, dy_matrix,
                 d_matrix: DMatrix,
                 b_alpha, mu):
        self.__k_matrix = KMatrix(k_values, dy_matrix, dx_matrix)
        self.__d_matrix = d_matrix
        self.__B_alpha = b_alpha
        self.__mu = mu
        self.__dx_matrix = dx_matrix
        self.__dy_matrix = dy_matrix

    def __getitem__(self, item):
        """
        returns an inter block value or in-block...
        :param item: [1.5, 3.5], not [1.5][3.5]
        :return:
        """
        # simplest case
        # major case - tuple
        if (type(item) == tuple) & (len(item) == 2):
            i, j = item
            if check_int(i) & check_half(j):
                out = self.__d_matrix[i, j] * self.__dx_matrix[i] * self.__k_matrix[i, j] / self.__B_alpha / self.__mu
                out /= (self.__dy_matrix(floor(j)) + self.__dy_matrix(ceil(j))) / 2
                return out

            if check_half(i) & check_int(j):
                out = self.__d_matrix[i, j] * self.__dy_matrix[i] * self.__k_matrix[i, j] / self.__B_alpha / self.__mu
                out /= (self.__dx_matrix(floor(j)) + self.__dx_matrix(ceil(j))) / 2
                return out
            # if for some reason not boundary condition
