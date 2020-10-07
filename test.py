from matrixes import KMatrix
from matrixes import DMatrix, TInterBlockMatrix, get_t_upd_matrix
import numpy as np

# test K_matrix
k_matrix = KMatrix(k_values=np.array([[1, 2, 3, 5, 7],
                                      [4, 5, 6, 1, 2],
                                      [9, 5, 6, 1, 4]]),
                   dy_matrix=np.array([7, 9, 7, 9, 7]),
                   dx_matrix=np.array([1, 2, 1])
                   )
assert k_matrix[0.5, 1] == 10 / 3
assert k_matrix[1, 1] == 5

d_matrix = DMatrix(d_values=np.array([[1, 2, 3, 5, 7],
                                      [4, 5, 6, 1, 2],
                                      [9, 5, 6, 1, 4]]),
                   )
assert d_matrix[0.5, 2] == 4.5
assert d_matrix[2, 2.5] == 3.5
assert d_matrix[1, 1] == 5


class TTestInter(TInterBlockMatrix):
    def __init__(self, shape=(3, 3)):
        self.shape = shape

    def __getitem__(self, item):
        i, j = item
        if (0 <= i) & (i <= self.shape[0] - 1) & (0 <= j) & (j <= self.shape[1] - 1):
            return 1
        else:
            return -2


t_int = TTestInter((3, 3))
t_upd = get_t_upd_matrix(t_int)
assert t_upd[0, 0] == -2, 't_upd matrix: corner'
assert t_upd[0, 1] == -1, 't_upd matrix: right neighbour'
assert t_upd[1, 0] == -1, 't_upd matrix: left neighbour element'
assert t_upd[0, 1] == -1, 't_upd matrix: right neighbour element'
assert t_upd[1, 1] == 1, 't_upd matrix: middle of the edge'
assert t_upd[4, 4] == 4, 't_upd matrix: in the center of matrix'
