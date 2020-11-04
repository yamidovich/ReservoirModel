import numpy as np
from math import floor
from properties import Constants
from interblock_matrixes import KMatrix, TInterBlockMatrix, DMatrix
import utils as u


def get_q_bound(t_matrix: TInterBlockMatrix, p_b) -> np.ndarray:
    nx, ny = t_matrix.shape
    out = np.zeros(nx * ny)
    if ny >= 1:
        for col_ind in range(ny):
            one_d = u.two_dim_index_to_one(0, col_ind, ny)
            out[one_d] += t_matrix[-0.5, col_ind] * p_b
            one_d = u.two_dim_index_to_one(nx - 1, col_ind, ny)
            out[one_d] += t_matrix[nx - 0.5, col_ind] * p_b
        for row_ind in range(nx):
            one_d = u.two_dim_index_to_one(row_ind, 0, ny)
            out[one_d] += t_matrix[row_ind, -0.5] * p_b
            one_d = u.two_dim_index_to_one(row_ind, ny - 1, ny)
            out[one_d] += t_matrix[row_ind, ny - 0.5] * p_b
    else:
        one_d = u.two_dim_index_to_one(0, 0, ny)
        out[one_d] += t_matrix[-0.5, 0] * p_b
        one_d = u.two_dim_index_to_one(nx - 1, 0, ny)
        out[one_d] += t_matrix[nx - 0.5, 0] * p_b

    return out.reshape((-1, 1))


def get_q_bound_const_p_well(j_matrix: np.ndarray, p_well):
    out = np.diag(j_matrix).copy().reshape((-1, 1))
    out *= p_well
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
        if ny > 1:
            out[d_i, d_i] += t[c_i[0], c_i[1] - 0.5]
            out[d_i, d_i] += t[c_i[0], c_i[1] + 0.5]

            if 0.5 <= c_i[1]:
                out[d_i, d_i - 1] = -1 * t[c_i[0], c_i[1] - 0.5]

            if c_i[1] <= (ny - 1) - 0.5:
                out[d_i, d_i + 1] = -1 * t[c_i[0], c_i[1] + 0.5]

        if 0.5 <= c_i[0]:
            out[d_i, d_i - ny] = -1 * t[c_i[0] - 0.5, c_i[1]]

        if c_i[0] <= (nx - 1) - 0.5:
            out[d_i, d_i + ny] = -1 * t[c_i[0] + 0.5, c_i[1]]

    return out


def get_b_p_w(porosity, consts: Constants = Constants()) -> np.ndarray:
    return np.diag(porosity.reshape(-1) * consts.c_t() / consts.b_w())


def get_q_well(index1d_q: dict, nx: int, ny: int, s_o, s_w) -> tuple:
    q_w = np.zeros((nx * ny))
    q_o = np.zeros((nx * ny))
    for key in index1d_q:
        sw = s_w[key]
        so = s_o[key]
        if s_w[key] < 0:
            sw = 0
        if s_o[key] < 0:
            so = 0
        if (so == 0) & (sw == 0):
            return q_w, q_o
        q_w[key] = index1d_q[key] * sw
        q_o[key] = index1d_q[key] * so

    return q_w.reshape((-1, 1)), q_o.reshape((-1, 1))


def get_q_well_total(index1d_q: dict, nx: int, ny: int) -> np.ndarray:
    out = np.zeros((nx * ny))
    for key in index1d_q:
        out[key] = index1d_q[key]
    return out.reshape((-1, 1))


def get_b_s_w(consts: Constants, porosity) -> np.ndarray:
    return np.diag(porosity.reshape(-1) * (consts.c_w() + consts.c_r()) / consts.b_w())


def get_b_s_o(consts: Constants, porosity) -> np.ndarray:
    return np.diag(porosity.reshape(-1) * (consts.c_o() + consts.c_r()) / consts.b_o())


def get_k_s_w(consts: Constants, k_with_rel: KMatrix) -> KMatrix:
    return k_with_rel / consts.mu_water() / consts.b_w()


def get_k_s_o(consts: Constants, k_with_rel: KMatrix) -> KMatrix:
    return k_with_rel / consts.mu_oil() / consts.b_o()


def get_k_tilde(consts: Constants, k_oil_with_rel: KMatrix, k_wat_with_rel: KMatrix) -> KMatrix:
    return get_k_s_o(consts, k_oil_with_rel) + get_k_s_w(consts, k_wat_with_rel)


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


def get_r_ref(i: int, j: int, depth: DMatrix, dx: np.array):
    out = 1. / dx[i] + np.pi / depth[i, j]
    out = 1. / out
    return out


def get_j_matrix_w(two_d_well_index_rw_scale: dict, nx: int, ny: int, const: Constants, k_matrix: KMatrix,
                   depth: DMatrix,
                   dx: np.array):
    out = np.zeros((nx * ny, nx * ny), dtype=float)
    for i, j in two_d_well_index_rw_scale:
        one_d_ind = u.two_dim_index_to_one(i, j, ny)
        r_ref = get_r_ref(i=i, j=j, depth=depth, dx=dx)
        r_well = two_d_well_index_rw_scale[(i, j)] * const.r_well()
        out[one_d_ind, one_d_ind] = 4 * np.pi / const.b_w() / const.mu_water() * k_matrix[i, j] * (r_ref * r_well) \
                                    / (r_ref + r_well)
    return out


def get_j_matrix_o(two_d_well_index_rw_scale: dict, nx: int, ny: int, const: Constants, k_matrix: KMatrix,
                   depth: DMatrix,
                   dx: np.array):
    out = np.zeros((nx * ny, nx * ny), dtype=float)
    for i, j in two_d_well_index_rw_scale:
        one_d_ind = u.two_dim_index_to_one(i, j, ny)
        r_ref = get_r_ref(i=i, j=j, depth=depth, dx=dx)
        r_well = two_d_well_index_rw_scale[(i, j)] * const.r_well()
        out[one_d_ind, one_d_ind] = 4 * np.pi / const.b_o() / const.mu_oil() * k_matrix[i, j] * (r_ref * r_well) \
                                    / (r_ref + r_well)
    return out


def get_q_const_p(j_matrix: np.ndarray, p_vec: np.ndarray, p_well: float, two_d_well_index_rw_scale: dict, ny: int):
    out = np.zeros((j_matrix.shape[0], 1))

    for i, j in two_d_well_index_rw_scale:
        diag = u.two_dim_index_to_one(i, j, ny)
        out[diag, 0] = (p_well - p_vec[diag, 0]) * j_matrix[diag, diag]
    return out
