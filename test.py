from matrixes import KMatrix
from matrixes import DMatrix
import numpy as np
# test K_matrix
k_matrix = KMatrix(k_values=np.array([[1, 2, 3, 5, 7],
                                      [4, 5, 6, 1, 2],
                                      [9, 5, 6, 1, 4]]),
                   dy_matrix=np.array([7, 9, 7, 9, 7]),
                   dx_matrix=np.array([1, 2, 1])
                   )
assert k_matrix[0.5, 1] == 10/3
assert k_matrix[1, 1] == 5

d_matrix = DMatrix(d_values=np.array([[1, 2, 3, 5, 7],
                                      [4, 5, 6, 1, 2],
                                      [9, 5, 6, 1, 4]]),
                   )
assert d_matrix[0.5, 2] == 4.5
assert d_matrix[2, 2.5] == 3.5
assert d_matrix[1, 1] == 5
