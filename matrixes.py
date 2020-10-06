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

    def __init__(self, k_values: np.ndarray, dy_matrix: np.array, dx_matrix: np.array):
        self.k = k_values
        self.dy = dy_matrix
        self.dx = dx_matrix
        self.shape = k_values.shape
        assert k_values.shape[0] == dx_matrix.shape[0]
        assert k_values.shape[1] == dy_matrix.shape[0]
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
            # if for bound condition reason not inter cell condition
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


def get_q_p(t_matrix: TInterBlockMatrix, p_b) -> np.ndarray:
    nx, ny = t_matrix.shape
    out = np.zeros(nx * ny)
    for col_ind in range(ny):
        out[col_ind] = 2 * t_matrix[-0.5, col_ind] * p_b
        out[nx * ny - ny + col_ind] = 2 * t_matrix[nx-0.5, col_ind] * p_b
    for row_ind in range(nx):
        out[ny * row_ind] = 2 * t_matrix[row_ind, -0.5] * p_b
        out[ny * (row_ind + 1) - 1] = 2 * t_matrix[ny-0.5, row_ind] * p_b
    return out


def get_b_matrix(depth: np.ndarray, dx: np.ndarray, dy: np.ndarray,
                 phi_matrix: np.ndarray, b_a: float) -> np.ndarray:
    return np.diag((depth * dx * dy * phi_matrix / b_a).reshpe(-1))


def get_t_upd_matrix(t: TInterBlockMatrix) -> np.ndarray:
    pass
