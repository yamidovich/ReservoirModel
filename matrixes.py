import numpy as np
from math import floor
from properties import Constants
from k_matrix import KMatrix
from t_interblock import TInterBlockMatrix


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


def get_b_p_w(consts: Constants, porosity) -> np.ndarray:
    return np.diag(porosity.reshape(-1) * consts.c_t() / consts.b_w())


def get_k_tilde(consts: Constants, k: KMatrix) -> KMatrix:
    return k * (consts.k_r_o() / consts.mu_oil() / consts.b_w() + consts.k_r_w() / consts.mu_water() / consts.b_w())


def get_q_well(index1d_q: dict, nx, ny) -> np.ndarray:
    out = np.zeros((nx * ny))
    for key in index1d_q:
        out[key] = index1d_q[key]
    return out


def get_b_s_w(consts: Constants, porosity) -> np.ndarray:
    return np.diag(porosity.reshape(-1) * (consts.c_w() + consts.c_r()) / consts.b_w())


def get_b_s_o(consts: Constants, porosity) -> np.ndarray:
    return np.diag(porosity.reshape(-1) * (consts.c_o() + consts.c_r()) / consts.b_o())


def get_k_s_w(consts: Constants, k: KMatrix) -> KMatrix:
    return k * consts.k_r_w() / consts.mu_water() / consts.b_w()


def get_k_s_o(consts: Constants, k: KMatrix) -> KMatrix:
    return k * consts.k_r_o() / consts.mu_water() / consts.b_o()
