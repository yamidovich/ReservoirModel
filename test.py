from matrixes import get_t_upd_matrix
from interblock_matrixes import TInterBlockMatrix, DMatrix, KMatrix
import numpy as np
import matrixes as ma

# test K_matrix
k_matrix = KMatrix(k_values=np.array([[1, 2, 3, 5, 7],
                                      [4, 5, 6, 1, 2],
                                      [9, 5, 6, 1, 4]]),
                   dy_matrix=np.array([7, 9, 7, 9, 7]),
                   dx_matrix=np.array([1, 2, 1])
                   )
assert k_matrix[0.5, 1] == 10 / 3
assert k_matrix[1, 1] == 5
# some operators
k_matrix = k_matrix / 5

assert k_matrix[1, 1] == 1
k_matrix *= 5
k_matrix *= 3
assert k_matrix[0.5, 1] == 10
k_matrix /= 3
# and the most complicated one - add
k_matrix1 = KMatrix(k_values=np.array([[1, 4, 3, 5, 7],
                                       [1, 5, 6, 1, 2],
                                       [9, 5, 4, 1, 7]]),
                    dy_matrix=np.array([7, 9, 7, 9, 7]),
                    dx_matrix=np.array([1, 2, 1])
                    )

k_matrix = k_matrix + k_matrix1
assert k_matrix[2, 1.5] == 16 / (9 / 5 + 7 / 4) + 16 / (9 / 5 + 7 / 6)

# some tests for depth
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


class KMatrTest(TInterBlockMatrix):
    def __init__(self, shape=(3, 3)):
        self.shape = shape

    def __getitem__(self, item):
        i, j = item
        if (0 <= i) & (i <= self.shape[0] - 1) & (0 <= j) & (j <= self.shape[1] - 1):
            return 1
        else:
            return 0


sh = (4, 4)
k_s_w = KMatrTest(shape=sh)
dx = np.ones(shape=sh[0])
dy = np.ones(shape=sh[1])
depth_m = DMatrix(np.ones(shape=sh))

t_k_s_w = TInterBlockMatrix(k_matrix=k_s_w,
                            dx_matrix=dx,
                            dy_matrix=dy,
                            d_matrix=depth_m,
                            boundary_condition='no_flux'
                            )
t_upd = get_t_upd_matrix(t_k_s_w)

assert t_upd[0, 0] == 2
assert t_upd[7, 7] == 3
assert t_upd[7, 6] == -1
assert t_upd[7, 3] == -1
assert t_upd[7, 11] == -1
assert t_upd[6, 6] == 4
assert t_upd[2, 2] == 3

q_tilde_p = ma.get_q_bound(t_k_s_w, 1)
assert q_tilde_p[0] == 0
assert q_tilde_p[1] == 0
assert q_tilde_p[5] == 0
assert q_tilde_p[3] == 0
assert q_tilde_p[2] == 0

t_k_s_w = TInterBlockMatrix(k_matrix=k_s_w,
                            dx_matrix=dx,
                            dy_matrix=dy,
                            d_matrix=depth_m,
                            boundary_condition='const_pressure'
                            )
t_upd = get_t_upd_matrix(t_k_s_w)
# print(t_upd)
assert t_upd[0, 0] == 6
assert t_upd[0, 1] == -1
assert t_upd[0, 4] == -1
assert t_upd[7, 7] == 5
assert t_upd[7, 6] == -1
assert t_upd[7, 3] == -1
assert t_upd[7, 11] == -1
assert t_upd[6, 6] == 4
assert t_upd[2, 2] == 5

q_tilde_p = ma.get_q_bound(t_k_s_w, 1)
assert q_tilde_p[0] == 4
assert q_tilde_p[1] == 2
assert q_tilde_p[5] == 0
assert q_tilde_p[3] == 4
assert q_tilde_p[2] == 2

