import numpy as np
import matrixes as ma
import interblock_matrixes as i_ma
from properties import Constants
import utils as u


class Env:
    def __init__(self, k_2d_matrix: np.ndarray, poir_2d_matrix: np.ndarray, depth_2d_matrix: np.ndarray,
                 satur_2d_matrix: np.ndarray,
                 const: Constants, well_positions: dict,
                 boundary_cond: dict = {'o': 'no_flux', 'p': 'const_pressure', 'w': 'const_pressure'}
                 ):
        self.boundary_cond = boundary_cond
        self.__const = const
        k_2d_matrix *= const.k_avg()
        depth_2d_matrix *= const.depth_avg()
        self.__nx, self.__ny = k_2d_matrix.shape
        # some asserts for shape
        assert k_2d_matrix.shape == poir_2d_matrix.shape
        assert poir_2d_matrix.shape == depth_2d_matrix.shape

        # initializing base properties
        dy = np.ones(self.__ny) * const.dy()
        dx = np.ones(self.__nx) * const.dx()

        self.__p_vec = np.ones((self.__nx * self.__ny)).reshape((-1, 1)) * const.p_0()
        self.__s_o_vec = satur_2d_matrix.reshape((-1, 1))
        self.__s_w_vec = (np.ones(satur_2d_matrix.shape) - satur_2d_matrix).reshape((-1, 1))

        v_matrix = np.diag(const.dx() * const.dy() * depth_2d_matrix.reshape(-1))
        v_matrix_inv = ma.inverse_diag(v_matrix)

        por_mat_diag = np.diag(poir_2d_matrix.reshape(-1))
        por_inv = ma.inverse_diag(por_mat_diag)
        # build base matrices for computation
        depth_m = i_ma.DMatrix(depth_2d_matrix)
        k_matrix_0 = i_ma.KMatrix(k_values=k_2d_matrix,
                                  dy_matrix=dy,
                                  dx_matrix=dx)
        # for pressure upd
        b_p_w = ma.get_b_p_w(poir_2d_matrix, const)
        self.__bpw_inv = ma.inverse_diag(b_p_w)
        k_tilde = ma.get_k_tilde(consts=const, k=k_matrix_0)
        self.__t_k_tilde = i_ma.TInterBlockMatrix(k_matrix=k_tilde,
                                                  dx_matrix=dx,
                                                  dy_matrix=dy,
                                                  d_matrix=depth_m,
                                                  boundary_condition=self.boundary_cond['p']
                                                  )
        t_upd_k_tilde = ma.get_t_upd_matrix(self.__t_k_tilde)
        self._inv_p_upd = np.linalg.inv(np.eye(self.__nx * self.__ny, dtype=float) +
                                        const.dt() * self.__bpw_inv.dot(t_upd_k_tilde))
        self.__b_rat = const.b_o() / const.b_w()
        # saturation of oil upd
        b_s_o = ma.get_b_s_o(consts=const, porosity=poir_2d_matrix)
        k_s_o = ma.get_k_s_o(consts=const, k=k_matrix_0)
        self.__t_k_s_o = i_ma.TInterBlockMatrix(k_matrix=k_s_o,
                                                dx_matrix=dx,
                                                dy_matrix=dy,
                                                d_matrix=depth_m,
                                                boundary_condition=self.boundary_cond['o']
                                                )
        self.__t_upd_k_s_o = ma.get_t_upd_matrix(self.__t_k_s_o)
        self.__tri_matr_o = ma.diagonal_multidot([por_inv, b_s_o, v_matrix_inv])
        # saturation of water upd
        b_s_w = ma.get_b_s_o(consts=const, porosity=poir_2d_matrix)
        k_s_w = ma.get_k_s_w(consts=const, k=k_matrix_0)
        self.__t_k_s_w = i_ma.TInterBlockMatrix(k_matrix=k_s_w,
                                                dx_matrix=dx,
                                                dy_matrix=dy,
                                                d_matrix=depth_m,
                                                boundary_condition=self.boundary_cond['w']
                                                )
        self.__t_upd_k_s_w = ma.get_t_upd_matrix(self.__t_k_s_w)
        self.__tri_matr_w = ma.diagonal_multidot([por_inv, b_s_w, v_matrix_inv])
        # wor both saturation
        self.__two_diag_dot = ma.diagonal_multidot([v_matrix_inv, por_inv])
        # wells and q
        self.__well_positions = {u.two_dim_index_to_one(i=k[0], j=k[1], ny=self.__ny):
                                     well_positions[k] for k in well_positions}
        self.__q_o = None
        self.__q_w = None

    def step(self):
        # q in wells
        self.__q_w, self.__q_o = ma.get_q_well(self.__well_positions,
                                               s_w=self.__s_w_vec, s_o=self.__s_o_vec,
                                               nx=self.__nx, ny=self.__ny)
        # boundary conditions
        q_tilde_p = ma.get_q_bound(self.__t_k_tilde, self.__const.p_0())
        q_tilde_w = ma.get_q_bound(self.__t_k_s_w, self.__const.p_0())
        q_tilde_o = ma.get_q_bound(self.__t_k_s_o, self.__const.p_0())
        # p upd
        p_vec_new = self.__p_vec + self.__const.dt() * self.__bpw_inv.dot(
            self.__b_rat * self.__q_o + self.__q_w + q_tilde_p)
        p_vec_new = self._inv_p_upd.dot(p_vec_new)
        # saturation upd
        # todo make saturation update implicit
        s_o_vec_new = self.__s_o_vec - (self.__const.c_o() + self.__const.c_r()) * (p_vec_new - self.__p_vec) * self.__s_o_vec
        s_o_vec_new += self.__const.dt() * self.__const.b_o() * self.__two_diag_dot.dot(
            -1 * self.__t_upd_k_s_o.dot(p_vec_new) + self.__q_o + q_tilde_o)

        s_w_vec_new = self.__s_w_vec - (self.__const.c_w() + self.__const.c_r()) * (p_vec_new - self.__p_vec) * self.__s_w_vec
        s_w_vec_new += self.__const.dt() * self.__const.b_w() * self.__two_diag_dot.dot(
            -1 * self.__t_upd_k_s_w.dot(p_vec_new) + self.__q_w + q_tilde_w)

        # upd self var
        self.__p_vec = p_vec_new
        self.__s_o_vec = s_o_vec_new
        self.__s_w_vec = s_w_vec_new
        # self.__s_o_vec = np.ones(s_w_vec_new.shape) - self.__s_w_vec
        # s_norm = self.__s_w_vec + self.__s_o_vec
        # self.__s_o_vec = s_o_vec_new / s_norm
        # self.__s_w_vec = s_w_vec_new / s_norm

    def s_w_as_2d(self):
        return self.__s_w_vec.reshape((self.__nx, self.__ny))

    def s_o_as_2d(self):
        return self.__s_o_vec.reshape((self.__nx, self.__ny))

    def p_as_2d(self):
        return self.__p_vec.reshape((self.__nx, self.__ny))

    def q_o_total(self):
        return -1 * self.__q_o.sum()

    def q_w_total(self):
        return -1 * self.__q_w.sum()
