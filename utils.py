from math import ceil, floor
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import display
from reservoir import Env
import numpy as np

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


def two_dim_index_to_one(i: int, j: int, ny: int) -> int:
    return ny * i + j


def one_d_index_to_two(one_d: int, ny: int):
    i = int(one_d / ny)
    j = one_d % ny
    return i, j


def plot_env_state_and_hist(n_it: int, upd_plot_freq: int, env: Env, q_w_rate: list, q_o_rate: list,
                            const_p_wells: dict, s_o_hist: dict, s_w_hist: dict, neibours: list,
                            saturation_matrix_0: np.ndarray, const, nx, ny):
    for i in range(n_it):
        env.step()

        if i % upd_plot_freq == 0:

            s_o = env.s_o_as_2d()
            s_w = env.s_w_as_2d()
            q_w_rate.append(env.q_w_total())
            q_o_rate.append(env.q_o_total())

            for key in const_p_wells:
                s_o_hist[key].append(s_o[key[0], key[1]])
                s_w_hist[key].append(s_w[key[0], key[1]])

            for key in neibours:
                s_o_hist[key].append(s_o[key[0], key[1]])
                s_w_hist[key].append(s_w[key[0], key[1]])

            display.clear_output(wait=True)
            f, ax = plt.subplots(nrows=4, ncols=2, figsize=(16, 16))
            f.tight_layout(pad=3.0)
            g1 = sns.heatmap(env.p_as_2d() / const.p_0(), ax=ax[0][0], cbar=True)  # , vmax=1, vmin=0.8)
            g1.set_title('Pressure / p_0')

            g2 = sns.heatmap(env.s_o_as_2d(), ax=ax[0][1], cbar=True)  # , vmax=0.5, vmin=0.3)
            g2.set_title('Saturation oil')

            g3 = sns.heatmap(s_w, ax=ax[1][0], cbar=True)  # , vmax=0.7, vmin=0.5)
            g3.set_title('Saturation water')
            satur_diff = saturation_matrix_0 - s_o
            for key in const_p_wells:
                satur_diff[key] = np.nan
                pass
            g4 = sns.heatmap(satur_diff, cbar=True, ax=ax[1][1])  # , vmax=1, vmin=0)
            g4.set_title('Saturation oil init - current\nwell cells are nan for better colors')

            # plot satur in every well
            for key in const_p_wells:
                ax[2][0].plot(s_o_hist[key], label=f'oil_satur, {key}')
                ax[2][0].plot(s_w_hist[key], label=f'water_satur, {key}')
            ax[2][0].set_title('Oil and water satur')
            ax[2][0].legend()

            ax[2][1].plot(q_o_rate, label='oil rate')
            ax[2][1].plot(q_w_rate, label='water rate')
            ax[2][1].set_title('Liquids rate')
            ax[2][1].legend()
            for key in neibours:
                ax[3][0].plot(s_o_hist[key], label=f'oil satur, {key}')
                ax[3][0].set_title('Oil satur for neibours')
            _ = sns.heatmap(np.ones((nx, ny)) - s_o - s_w, cbar=True, ax=ax[3][1])  # , vmax=1, vmin=0)
            ax[3][1].set_title('1 - S_o - S_w')
            plt.show()
