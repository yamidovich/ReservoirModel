import numpy as np
import utils as u
from d_matrix import DMatrix
from k_matrix import KMatrix
from math import ceil, floor


class TInterBlockMatrix:
    """
    class for matrix of depth. we got inter block values, but what do we have on boundaries...
    so we have to call [1, 4.5] to get proper inter block value
    """

    def __init__(self,
                 k_matrix: KMatrix,
                 dx_matrix: np.array,
                 dy_matrix: np.array,
                 d_matrix: DMatrix):
        self.__k_matrix = k_matrix
        self.__d_matrix = d_matrix
        self.__dx_matrix = dx_matrix
        self.__dy_matrix = dy_matrix
        self.shape = k_matrix.shape

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
            nx, ny = self.__k_matrix.shape
            # here are bounds
            if (i == -0.5) & (j <= ny - 1) & (0 <= j) & u.check_int(j):
                i = 0
                out = self.__d_matrix[i, j] * self.__dy_matrix[floor(i)] * self.__k_matrix[i, j]
                out /= (self.__dx_matrix[floor(j)] + self.__dx_matrix[ceil(j)]) / 2
                return 2 * out
            # one of bound
            if (i == nx - 0.5) & (j <= ny - 1) & (0 <= j) & u.check_int(j):
                i = nx - 1
                out = self.__d_matrix[i, j] * self.__dy_matrix[floor(i)] * self.__k_matrix[i, j]
                out /= (self.__dx_matrix[floor(j)] + self.__dx_matrix[ceil(j)]) / 2
                return 2 * out

            # other 2 line bounds
            if (j == -0.5) & (i <= nx - 1) & (0 <= i) & u.check_int(i):
                j = 0
                out = self.__d_matrix[i, j] * self.__dx_matrix[floor(i)] * self.__k_matrix[i, j]
                out /= (self.__dy_matrix[floor(j)] + self.__dy_matrix[ceil(j)]) / 2
                return 2 * out
            # bound
            if (j == ny - 0.5) & (i <= nx - 1) & (0 <= i) & u.check_int(i):
                j = ny - 1
                out = self.__d_matrix[i, j] * self.__dx_matrix[floor(i)] * self.__k_matrix[i, j]
                out /= (self.__dy_matrix[floor(j)] + self.__dy_matrix[ceil(j)]) / 2
                return 2 * out

            # major cases
            if u.check_int(i) & u.check_half(j):
                out = self.__d_matrix[i, j] * self.__dx_matrix[floor(i)] * self.__k_matrix[i, j]
                out /= (self.__dy_matrix[floor(j)] + self.__dy_matrix[ceil(j)]) / 2
                return out

            elif u.check_half(i) & u.check_int(j):
                out = self.__d_matrix[i, j] * self.__dy_matrix[floor(i)] * self.__k_matrix[i, j]
                out /= (self.__dx_matrix[floor(j)] + self.__dx_matrix[ceil(j)]) / 2
                return out
            else:
                assert False, "wrong index, not int + int and a half-like int"
