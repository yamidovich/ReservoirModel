from math import ceil, floor
import numpy as np
import utils as u


class DMatrix:
    """
    class for matrix of depth. we got inter block values, but what do we have on boundaries...
    so we have to call [1, 4.5] to get proper inter block value
    """

    def __init__(self, d_values):
        self.__d = d_values
        self.shape = d_values.shape

    def __getitem__(self, item):
        """
        returns an inter block value or in-block...
        :param item: [1.5, 3.5], not [1.5][3.5]
        :return:
        """
        # simplest case
        if u.check_if_numerical(self.__d):
            return self.__d
        # major case - tuple
        if (type(item) == tuple) & (len(item) == 2):
            i, j = item

            # bound
            if i < 0:
                return self.__d[0, j]
            if i >= self.__d.shape[0] - 0.6:
                return self.__d[self.__d.shape[0] - 1, j]
            if j < 0:
                return self.__d[i, 0]
            if j >= self.__d.shape[1] - 0.6:
                return self.__d[i, self.__d.shape[1] - 1]

            if u.check_int(i) & u.check_half(j):
                out = self.__d[floor(i), floor(j)] + self.__d[floor(i), ceil(j)]
                out *= 0.5
                return out

            if u.check_half(i) & u.check_int(j):
                out = self.__d[floor(i), floor(j)] + self.__d[ceil(i), floor(j)]
                out *= 0.5
                return out
            # if for some reason not boundary condition
            if u.check_int(i) & u.check_int(j):
                return self.__d[item]

        elif type(item) == int:
            return self.__d[item]


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
        self.shape = self.k[0].shape
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
            # bound
            if i < 0:
                return self.k[num][0, j]
            if i >= self.shape[0] - 0.6:
                return self.k[num][self.shape[0] - 1, j]
            if j < 0:
                return self.k[num][i, 0]
            if j >= self.shape[1] - 0.6:
                return self.k[num][i, self.shape[1] - 1]

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


class TInterBlockMatrix:
    """
    class for matrix of depth. we got inter block values, but what do we have on boundaries...
    so we have to call [1, 4.5] to get proper inter block value
    """

    def __init__(self,
                 k_matrix: KMatrix,
                 dx_matrix: np.array,
                 dy_matrix: np.array,
                 d_matrix: DMatrix,
                 boundary_condition='const_pressure'):
        self.__k_matrix = k_matrix
        self.__d_matrix = d_matrix
        self.__dx_matrix = dx_matrix
        self.__dy_matrix = dy_matrix
        self.shape = k_matrix.shape
        self.__boundary_condition = boundary_condition

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
            if self.__boundary_condition == 'const_pressure':
                if (i == -0.5) & (j <= ny - 1) & (0 <= j) & u.check_int(j):
                    i = 0
                    out = self.__d_matrix[i, j] * self.__dy_matrix[floor(i)] * self.__k_matrix[i, j]
                    out /= (self.__dx_matrix[floor(j)] + self.__dx_matrix[ceil(j)]) / 2
                    return 2 * out
                    # return 0
                # one of bound
                if (i == nx - 0.5) & (j <= ny - 1) & (0 <= j) & u.check_int(j):
                    i = nx - 1
                    out = self.__d_matrix[i, j] * self.__dy_matrix[floor(i)] * self.__k_matrix[i, j]
                    out /= (self.__dx_matrix[floor(j)] + self.__dx_matrix[ceil(j)]) / 2
                    return 2 * out
                    # return 0

                # other 2 line bounds
                if (j == -0.5) & (i <= nx - 1) & (0 <= i) & u.check_int(i):
                    j = 0
                    out = self.__d_matrix[i, j] * self.__dx_matrix[floor(i)] * self.__k_matrix[i, j]
                    out /= (self.__dy_matrix[floor(j)] + self.__dy_matrix[ceil(j)]) / 2
                    return 2 * out
                    # return 0
                # bound
                if (j == ny - 0.5) & (i <= nx - 1) & (0 <= i) & u.check_int(i):
                    j = ny - 1
                    out = self.__d_matrix[i, j] * self.__dx_matrix[floor(i)] * self.__k_matrix[i, j]
                    out /= (self.__dy_matrix[floor(j)] + self.__dy_matrix[ceil(j)]) / 2
                    return 2 * out
                    # return 0
            elif self.__boundary_condition == 'no_flux':
                if (i == -0.5) & (j <= ny - 1) & (0 <= j) & u.check_int(j):
                    return 0
                # one of bound
                if (i == nx - 0.5) & (j <= ny - 1) & (0 <= j) & u.check_int(j):
                    return 0

                # other 2 line bounds
                if (j == -0.5) & (i <= nx - 1) & (0 <= i) & u.check_int(i):
                    return 0
                # bound
                if (j == ny - 0.5) & (i <= nx - 1) & (0 <= i) & u.check_int(i):
                    return 0

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
