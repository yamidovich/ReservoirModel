import numpy as np
import matrixes as ma
import interblock_matrixes as i_ma
from properties import Constants
import utils as u


class Env:
    def __init__(self, k_2d_matrix: np.ndarray, poir_2d_matrix: np.ndarray, depth_2d_matrix: np.ndarray,
                 satur_2d_matrix: np.ndarray,
                 const: Constants, well_positions: dict,
                 two_d_well_index_rw_scale: dict,
                 boundary_cond: str = 'const_pressure'
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
        self.__dy = np.ones(self.__ny) * const.dy()
        self.__dx = np.ones(self.__nx) * const.dx()

        self.__p_vec = np.ones((self.__nx * self.__ny)).reshape((-1, 1)) * const.p_0()
        self.__s_o_vec = satur_2d_matrix.reshape((-1, 1))
        self.__s_w_vec = (np.ones(satur_2d_matrix.shape) - satur_2d_matrix).reshape((-1, 1))

        v_matrix = np.diag(const.dx() * const.dy() * depth_2d_matrix.reshape(-1))
        v_matrix_inv = ma.inverse_diag(v_matrix)

        por_mat_diag = np.diag(poir_2d_matrix.reshape(-1))
        por_inv = ma.inverse_diag(por_mat_diag)
        # build base matrices for computation
        self.__depth_m = i_ma.DMatrix(depth_2d_matrix)
        self.__k_matrix_0 = i_ma.KMatrix(k_values=k_2d_matrix,
                                         dy_matrix=self.__dy,
                                         dx_matrix=self.__dx)
        # for pressure upd
        b_p_w = ma.get_b_p_w(poir_2d_matrix, const)
        self.__bpw_inv = ma.inverse_diag(b_p_w)
        k_tilde = ma.get_k_tilde(consts=const, k=self.__k_matrix_0)
        self.__t_k_tilde = i_ma.TInterBlockMatrix(k_matrix=k_tilde,
                                                  dx_matrix=self.__dx,
                                                  dy_matrix=self.__dy,
                                                  d_matrix=self.__depth_m,
                                                  boundary_condition=self.boundary_cond
                                                  )
        self.__t_upd_k_tilde = ma.get_t_upd_matrix(self.__t_k_tilde)
        self._inv_p_upd = np.linalg.inv(np.eye(self.__nx * self.__ny, dtype=float) +
                                        const.dt() * self.__bpw_inv.dot(self.__t_upd_k_tilde))
        self.__b_rat = const.b_o() / const.b_w()
        # saturation of oil upd
        b_s_o = ma.get_b_s_o(consts=const, porosity=poir_2d_matrix)
        k_s_o = ma.get_k_s_o(consts=const, k=self.__k_matrix_0)
        self.__t_k_s_o = i_ma.TInterBlockMatrix(k_matrix=k_s_o,
                                                dx_matrix=self.__dx,
                                                dy_matrix=self.__dy,
                                                d_matrix=self.__depth_m,
                                                boundary_condition=self.boundary_cond
                                                )
        self.__t_upd_k_s_o = ma.get_t_upd_matrix(self.__t_k_s_o)
        self.__tri_matr_o = ma.diagonal_multidot([por_inv, b_s_o, v_matrix_inv])
        # saturation of water upd
        b_s_w = ma.get_b_s_o(consts=const, porosity=poir_2d_matrix)
        k_s_w = ma.get_k_s_w(consts=const, k=self.__k_matrix_0)
        self.__t_k_s_w = i_ma.TInterBlockMatrix(k_matrix=k_s_w,
                                                dx_matrix=self.__dx,
                                                dy_matrix=self.__dy,
                                                d_matrix=self.__depth_m,
                                                boundary_condition=self.boundary_cond
                                                )
        self.__t_upd_k_s_w = ma.get_t_upd_matrix(self.__t_k_s_w)
        self.__tri_matr_w = ma.diagonal_multidot([por_inv, b_s_w, v_matrix_inv])
        # wor both saturation
        self.__two_diag_dot = ma.diagonal_multidot([v_matrix_inv, por_inv])
        # wells and q
        self.__two_d_well_index_rw_scale = two_d_well_index_rw_scale
        self.__well_positions_cr = {u.two_dim_index_to_one(i=k[0], j=k[1], ny=self.__ny):
                                        well_positions[k] for k in well_positions}
        self.__q_o = np.zeros((self.__nx * self.__ny, 1))
        self.__q_w = np.zeros((self.__nx * self.__ny, 1))

        self.__p_well = self.__const.p_0() - self.__const.delta_p()

    def step(self):
        # q in wells
        # self.__q_w, self.__q_o = ma.get_q_well(self.__well_positions_cr, s_w=self.__s_w_vec, s_o=self.__s_o_vec,
        #                                        nx=self.__nx, ny=self.__ny)
        self.__q_o = ma.get_q_well_total(self.__well_positions_cr, nx=self.__nx, ny=self.__ny) * self.__s_o_vec
        self.__q_w = ma.get_q_well_total(self.__well_positions_cr, nx=self.__nx, ny=self.__ny) * self.__s_w_vec
        # boundary conditions
        q_tilde_p = ma.get_q_bound(self.__t_k_tilde, self.__const.p_0())
        # q_tilde_w = ma.get_q_bound(self.__t_k_s_w, self.__const.p_0())
        q_tilde_o = ma.get_q_bound(self.__t_k_s_o, self.__const.p_0())
        # const pressure well
        j_w = ma.get_j_matrix_w(self.__two_d_well_index_rw_scale, nx=self.__nx, ny=self.__ny,
                                const=self.__const, k_matrix=self.__k_matrix_0,
                                depth=self.__depth_m, dx=self.__dx)
        j_o = ma.get_j_matrix_w(self.__two_d_well_index_rw_scale, nx=self.__nx, ny=self.__ny,
                                const=self.__const, k_matrix=self.__k_matrix_0,
                                depth=self.__depth_m, dx=self.__dx)
        j_w *= self.__s_w_vec
        j_o *= self.__s_o_vec
        q_tilde_p_cpo = ma.get_q_bound_const_p_well(j_o, self.__p_well)
        q_tilde_p_cpw = ma.get_q_bound_const_p_well(j_w, self.__p_well)
        # pressure upd
        inv_p_upd = np.eye(self.__nx * self.__ny, dtype=float) \
                    + self.__const.dt() * self.__bpw_inv.dot(
            self.__t_upd_k_tilde + j_w + j_o * self.__b_rat)

        inv_p_upd = np.linalg.inv(inv_p_upd)
        p_vec_new = self.__p_vec + \
                    self.__const.dt() * self.__bpw_inv.dot(
            self.__b_rat * self.__q_o * self.__s_o_vec + self.__q_w * self.__s_w_vec + q_tilde_p + q_tilde_p_cpo * self.__b_rat + q_tilde_p_cpw)
        p_vec_new = inv_p_upd.dot(p_vec_new)
        # prepare to saturation upd
        q_o_cp = ma.get_q_const_p(j_matrix=j_o, p_vec=p_vec_new, p_well=self.__p_well,
                                  two_d_well_index_rw_scale=self.__two_d_well_index_rw_scale, ny=self.__ny)
        self.__q_o += q_o_cp
        q_w_cp = ma.get_q_const_p(j_matrix=j_w, p_vec=p_vec_new, p_well=self.__p_well,
                                  two_d_well_index_rw_scale=self.__two_d_well_index_rw_scale, ny=self.__ny)
        self.__q_w += q_w_cp
        # upd saturation
        self.__q_o /= self.__s_o_vec

        s_o_div = np.ones(self.__s_o_vec.shape) + (self.__const.c_o() + self.__const.c_r()) * (p_vec_new - self.__p_vec)
        s_o_div -= self.__const.dt() * self.__const.b_o() * self.__two_diag_dot.dot(self.__q_o)

        s_o_vec_new = self.__s_o_vec + self.__const.dt() * self.__const.b_o() * self.__two_diag_dot.dot(
            -1 * self.__t_upd_k_s_o.dot(p_vec_new) + q_tilde_o)
        s_o_vec_new /= s_o_div
        self.__q_o *= s_o_vec_new
        # s_w_div = np.ones(self.__s_w_vec.shape) + (self.__const.c_w() + self.__const.c_r()) * (p_vec_new - self.__p_vec)
        # s_w_div -= self.__const.dt() * self.__const.b_w() * self.__two_diag_dot.dot(self.__q_w)
        # s_w_vec_new = self.__s_w_vec + self.__const.dt() * self.__const.b_w() * self.__two_diag_dot.dot(
        #     -1 * self.__t_upd_k_s_w.dot(p_vec_new) + q_tilde_w)
        # s_w_vec_new /= s_w_div

        # upd self var
        self.__p_vec = p_vec_new
        self.__s_o_vec = s_o_vec_new
        # self.__s_w_vec = s_w_vec_new
        self.__s_w_vec = np.ones(s_o_vec_new.shape) - self.__s_o_vec

    def s_w_as_2d(self):
        return self.__s_w_vec.reshape((self.__nx, self.__ny))

    def s_o_as_2d(self):
        return self.__s_o_vec.reshape((self.__nx, self.__ny))

    def p_as_2d(self):
        return self.__p_vec.reshape((self.__nx, self.__ny))

    def q_o_total(self):
        out = -1 * self.__q_o.sum()
        return out

    def q_w_total(self):
        return -1 * self.__q_w.sum()
