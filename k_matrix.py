import numpy as np
import utils as u
from math import floor, ceil


class KMatrix:
    """
    class for matrix k. we got inter block values, but what do we have on boundaries...
    so we have to call [1.5, 4.5] to get proper inter block value
    """

    def __init__(self, k_values, dy_matrix: np.array, dx_matrix: np.array):
        if type(k_values) == list:
            self.k = k_values
        elif type(k_values) == np.ndarray:
            self.k = [k_values]
        else:
            assert False, 'k_values are list of np.ndarray of one of them'
        self.dy = dy_matrix
        self.dx = dx_matrix
        self.__nums = len(self.k)
        self.shape = k_values[0].shape
        assert self.k[0].shape[0] == dx_matrix.shape[0]
        assert self.k[0].shape[1] == dy_matrix.shape[0]
        # what if case is simple and no matrix for k and dx, dy needed and won't be called
        # but if k is a matrix and grid dx is a constant step -
        if u.check_if_numerical(self.dx) & u.check_if_numerical(self.dy):
            self.dx = np.ones(self.k[0].shape[0]) * self.dx
            self.dy = np.ones(self.k[0].shape[1]) * self.dy

    def __truediv__(self, other):
        __k_values = None
        if type(self.k) == list:
            __k_values = [_k / other for _k in self.k]
        elif type(self.k) == np.ndarray:
            __k_values = self.k / other
        else:
            assert False, 'k_values are list of np.ndarray of one of them'
        out = KMatrix(k_values=__k_values, dx_matrix=self.dx, dy_matrix=self.dy)
        return out

    def __mul__(self, other):
        __k_values = None
        if type(self.k) == list:
            __k_values = [_k * other for _k in self.k]
        elif type(self.k) == np.ndarray:
            __k_values = self.k * other
        else:
            assert False, 'k_values are list of np.ndarray of one of them'
        out = KMatrix(k_values=__k_values, dx_matrix=self.dx, dy_matrix=self.dy)
        return out

    def __getitem_part__(self, item, num):
        """
        returns an inter block value or in-block...
        :param item: [1.5, 3.5], not [1.5][3.5]
        :return:
        """
        # major case - tuple
        if (type(item) == tuple) & (len(item) == 2):
            i, j = item

            if u.check_int(i) & u.check_half(j):
                out = self.dy[floor(j)] / self.k[num][floor(i), floor(j)]
                out += self.dy[ceil(j)] / self.k[num][floor(i), ceil(j)]
                out = 1 / out
                out *= self.dy[floor(j)] + self.dy[ceil(j)]
                return out

            if u.check_half(i) & u.check_int(j):
                out = self.dx[floor(i)] / self.k[num][floor(i), floor(j)]
                out += self.dx[ceil(i)] / self.k[num][ceil(i), floor(j)]
                out = 1 / out
                out *= self.dx[floor(i)] + self.dx[ceil(i)]
                return out
            # if for bound condition reason not inter cell condition
            if u.check_int(i) & u.check_int(j):
                return self.k[num][item]

        elif type(item) == int:
            return self.k[num][item]

    def __getitem__(self, item):
        """
        returns an inter block value or in-block...
        :param item: [1.5, 3.5], not [1.5][3.5]
        :return:
        """
        out = 0
        for n in range(self.__nums):
            out += self.__getitem_part__(item=item, num=n)
        return out

    def __add__(self, other):
        comparison = self.dx == other.dx
        equal_arrays = comparison.all()
        assert equal_arrays

        comparison = self.dy == other.dy
        equal_arrays = comparison.all()
        assert equal_arrays
        return KMatrix(k_values=self.k + other.k, dx_matrix=self.dx, dy_matrix=self.dy)
