import numpy as np
from math import floor
from properties import Constants
from k_matrix import KMatrix
from t_interblock import TInterBlockMatrix


def get_q_bound(t_matrix: TInterBlockMatrix, p_b) -> np.ndarray:
    nx, ny = t_matrix.shape
    out = np.zeros(nx * ny)
    for col_ind in range(ny):
        out[col_ind] += 1 * t_matrix[-0.5, col_ind] * p_b
        out[nx * ny - ny + col_ind] += 2 * t_matrix[nx-0.5, col_ind] * p_b
    for row_ind in range(nx):
        out[ny * row_ind] = 1 * t_matrix[row_ind, -0.5] * p_b
        out[ny * (row_ind + 1) - 1] += 2 * t_matrix[ny-0.5, row_ind] * p_b
    return out


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

        if 0 <= c_i[1] - 0.5:
            out[d_i, d_i - 1] = -1 * t[c_i[0], c_i[1] - 0.5]

        if c_i[1] < ny-1:
            out[d_i, d_i + 1] = -1 * t[c_i[0], c_i[1] + 0.5]

        if 0 <= c_i[0] - 0.5:
            out[d_i, d_i - ny] = -1 * t[c_i[0] - 0.5, c_i[1]]

        if c_i[0] < nx - 1:
            out[d_i, d_i + ny] = -1 * t[c_i[0] + 0.5, c_i[1]]

    return out


def get_b_p_w(porosity, consts: Constants = Constants()) -> np.ndarray:
    return np.diag(porosity.reshape(-1) * consts.c_t() / consts.b_w())


def get_k_tilde(consts: Constants, k: KMatrix) -> KMatrix:
    return k * (consts.k_r_o() / consts.mu_oil() / consts.b_w() + consts.k_r_w() / consts.mu_water() / consts.b_w())


def get_q_well(index1d_q: dict, nx: int, ny: int, s_o, s_w) -> tuple:
    q_w = np.zeros((nx * ny))
    q_o = np.zeros((nx * ny))
    for key in index1d_q:
        q_w[key] = index1d_q[key] * s_w[key] / (s_w[key] + s_o[key])
        q_o[key] = index1d_q[key] * s_o[key] / (s_w[key] + s_o[key])

    return q_w, q_o


def get_b_s_w(consts: Constants, porosity) -> np.ndarray:
    return np.diag(porosity.reshape(-1) * (consts.c_w() + consts.c_r()) / consts.b_w())


def get_b_s_o(consts: Constants, porosity) -> np.ndarray:
    return np.diag(porosity.reshape(-1) * (consts.c_o() + consts.c_r()) / consts.b_o())


def get_k_s_w(consts: Constants, k: KMatrix) -> KMatrix:
    return k * consts.k_r_w() / consts.mu_water() / consts.b_w()


def get_k_s_o(consts: Constants, k: KMatrix) -> KMatrix:
    return k * consts.k_r_o() / consts.mu_water() / consts.b_o()


def inverse_diag(x: np.ndarray):
    assert x.shape[0] == x.shape[1]
    out = x.copy()
    for i in range(x.shape[0]):
        out[i, i] = 1. / x[i, i]
    return out


def diagonal_multidot(maxtrixes: list):
    out = np.zeros(maxtrixes[0].shape)
    np.fill_diagonal(out, 1)
    for i in range(out.shape[0]):
        for m in maxtrixes:
            out[i, i] *= m[i, i]
    return out
