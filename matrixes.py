import numpy as np
from math import floor, ceil


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
        if check_if_numerical(self.dx) & check_if_numerical(self.dy):
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

            if check_int(i) & check_half(j):
                out = self.dy[floor(j)] / self.k[num][floor(i), floor(j)]
                out += self.dy[ceil(j)] / self.k[num][floor(i), ceil(j)]
                out = 1 / out
                out *= self.dy[floor(j)] + self.dy[ceil(j)]
                return out

            if check_half(i) & check_int(j):
                out = self.dx[floor(i)] / self.k[num][floor(i), floor(j)]
                out += self.dx[ceil(i)] / self.k[num][ceil(i), floor(j)]
                out = 1 / out
                out *= self.dx[floor(i)] + self.dx[ceil(i)]
                return out
            # if for bound condition reason not inter cell condition
            if check_int(i) & check_int(j):
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
                 dx_matrix: np.array,
                 dy_matrix: np.array,
                 d_matrix: DMatrix):
        self.__k_matrix = KMatrix(k_values, dy_matrix, dx_matrix)
        self.__d_matrix = d_matrix
        self.__dx_matrix = dx_matrix
        self.__dy_matrix = dy_matrix
        assert k_values.shape[0] == dx_matrix.shape[0], 'input shape issue, k_values and dx'
        assert k_values.shape[1] == dy_matrix.shape[0], 'input shape issue k_values and dy'
        assert k_values.shape == d_matrix.shape, 'input shape issue k values and depth'
        self.shape = k_values.shape

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
            if (i == -0.5) & (j <= ny - 1) & (0 <= j) & check_int(j):
                i = 0
                out = self.__d_matrix[i, j] * self.__dy_matrix[floor(i)] * self.__k_matrix[i, j]
                out /= (self.__dx_matrix(floor(j)) + self.__dx_matrix(ceil(j))) / 2
                return 2 * out
            # one of bound
            if (i == nx - 0.5) & (j <= ny - 1) & (0 <= j) & check_int(j):
                i = nx - 1
                out = self.__d_matrix[i, j] * self.__dy_matrix[floor(i)] * self.__k_matrix[i, j]
                out /= (self.__dx_matrix(floor(j)) + self.__dx_matrix(ceil(j))) / 2
                return 2 * out

            # other 2 line bounds
            if (j == -0.5) & (i <= nx - 1) & (0 <= i) & check_int(i):
                j = 0
                out = self.__d_matrix[i, j] * self.__dx_matrix[floor(i)] * self.__k_matrix[i, j]
                out /= (self.__dy_matrix(floor(j)) + self.__dy_matrix(ceil(j))) / 2
                return 2 * out
            # bound
            if (j == ny - 0.5) & (j <= ny - 1) & (0 <= j) & check_int(j):
                j = ny - 1
                out = self.__d_matrix[i, j] * self.__dx_matrix[floor(i)] * self.__k_matrix[i, j]
                out /= (self.__dy_matrix(floor(j)) + self.__dy_matrix(ceil(j))) / 2
                return 2 * out

            # major cases
            if check_int(i) & check_half(j):
                out = self.__d_matrix[i, j] * self.__dx_matrix[floor(i)] * self.__k_matrix[i, j]
                out /= (self.__dy_matrix(floor(j)) + self.__dy_matrix(ceil(j))) / 2
                return out

            elif check_half(i) & check_int(j):
                out = self.__d_matrix[i, j] * self.__dy_matrix[floor(i)] * self.__k_matrix[i, j]
                out /= (self.__dx_matrix(floor(j)) + self.__dx_matrix(ceil(j))) / 2
                return out
            else:
                assert False, "wrong index, not int + int and a half-like int"


def get_q_bound(t_matrix: TInterBlockMatrix, p_b) -> np.ndarray:
    nx, ny = t_matrix.shape
    out = np.zeros(nx * ny)
    for col_ind in range(ny):
        out[col_ind] += 2 * t_matrix[-0.5, col_ind] * p_b
        out[nx * ny - ny + col_ind] += 2 * t_matrix[nx-0.5, col_ind] * p_b
    for row_ind in range(nx):
        out[ny * row_ind] += 2 * t_matrix[row_ind, -0.5] * p_b
        out[ny * (row_ind + 1) - 1] += 2 * t_matrix[ny-0.5, row_ind] * p_b
    return out


def get_b_matrix(depth: np.ndarray, dx: np.ndarray, dy: np.ndarray,
                 phi_matrix: np.ndarray, b_a: float) -> np.ndarray:
    return np.diag((depth * dx * dy * phi_matrix / b_a).reshpe(-1))


def two_dim_index_to_one(i: int, j: int, ny: int) -> int:
    return ny * i + j


def one_dim_index_to_two(m: int, ny: int) -> tuple:
    return floor(m / ny), m % ny


def get_t_upd_matrix(t: TInterBlockMatrix) -> np.ndarray:
    nx, ny = t.shape
    out = np.zeros((nx * ny, nx * ny))
    for d_i in range(nx * ny):
        c_i = one_dim_index_to_two(m=d_i, ny=ny)
        out[d_i, d_i] += t[c_i[0] + 0.5, c_i[1]]
        out[d_i, d_i] += t[c_i[0] - 0.5, c_i[1]]
        out[d_i, d_i] += t[c_i[0], c_i[1] - 0.5]
        out[d_i, d_i] += t[c_i[0], c_i[1] + 0.5]

        if 0 <= d_i - 1 < ny * nx:
            out[d_i, d_i - 1] = -1 * t[c_i[0], c_i[1] - 0.5]

        if 0 <= d_i + 1 < ny * nx:
            out[d_i, d_i + 1] = -1 * t[c_i[0], c_i[1] + 0.5]

        if 0 <= d_i - ny < ny * nx:
            out[d_i, d_i - ny] = -1 * t[c_i[0] - 0.5, c_i[1]]

        if 0 <= d_i + ny < ny * nx:
            out[d_i, d_i + ny] = -1 * t[c_i[0] + 0.5, c_i[1]]

    return out
