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


def plot_stuff_iter(q_w_rate: list, q_o_rate: list, const_p_wells_keys: list, s_o_hist: dict, s_w_hist: dict,
                    s_o_matrix_history: list, neibours: list,
                    p_matrix_history: list, i: int = -1
                    ):
    display.clear_output(wait=True)
    f, ax = plt.subplots(nrows=4, ncols=2, figsize=(16, 16))
    f.tight_layout(pad=3.0)
    g1 = sns.heatmap(p_matrix_history[i], ax=ax[0][0], cbar=True)  # , vmax=1, vmin=0.8)
    g1.set_title('Pressure / p_0')

    g2 = sns.heatmap(s_o_matrix_history[i], ax=ax[0][1], cbar=True)  # , vmax=0.5, vmin=0.3)
    g2.set_title('Saturation oil')
    s_w = np.ones(s_o_matrix_history[i].shape) - s_o_matrix_history[-1]
    g3 = sns.heatmap(s_w, ax=ax[1][0], cbar=True)  # , vmax=0.7, vmin=0.5)
    g3.set_title('Saturation water')
    satur_diff = s_o_matrix_history[0] - s_o_matrix_history[i]
    for key in const_p_wells_keys:
        satur_diff[key] = np.nan
        pass
    g4 = sns.heatmap(satur_diff, cbar=True, ax=ax[1][1])  # , vmax=1, vmin=0)
    g4.set_title('Saturation oil init - current\nwell cells are nan for better colors')

    # plot satur in every well
    k = 0
    for key in const_p_wells_keys:
        k = key
        ax[2][0].plot(s_o_hist[key][:i], label=f'oil_satur, {key}')
        ax[2][0].plot(s_w_hist[key][:i], label=f'water_satur, {key}')
    ax[2][0].set_title('Oil and water satur')
    ax[2][0].legend()
    ax[2][0].hlines(0, xmin=0, xmax=i-1 if i > 0 else 0, linestyles='dashed')
    ax[2][0].hlines(1, xmin=0, xmax=i-1 if i > 0 else 0, linestyles='dashed')

    ax[2][1].plot(q_o_rate[:i], label='oil rate')
    ax[2][1].plot(q_w_rate[:i], label='water rate')
    ax[2][1].set_title('Liquids rate')
    ax[2][1].legend()
    for key in neibours:
        ax[3][0].plot(s_o_hist[key][:i], label=f'oil satur, {key}')
        ax[3][0].set_title('Oil satur for neibours')
    _ = sns.heatmap(np.ones(s_o_matrix_history[i].shape) - s_o_matrix_history[-1] - s_w,
                    cbar=True, ax=ax[3][1])  # , vmax=1, vmin=0)
    ax[3][1].set_title('1 - S_o - S_w')
    plt.show()


def plot_env_state_and_hist(n_it: int, upd_plot_freq: int, env: Env, q_w_rate: list, q_o_rate: list,
                            const_p_wells: dict, s_o_hist: dict, s_w_hist: dict, s_o_matrix_history: list,
                            p_matrix_history: list, neibours: list, const):
    for i in range(n_it):
        env.step()

        if i % upd_plot_freq == 0:

            s_o = env.s_o_as_2d()
            s_w = env.s_w_as_2d()
            q_w_rate.append(env.q_w_total())
            q_o_rate.append(env.q_o_total())
            s_o_matrix_history.append(s_o)
            p_matrix_history.append(env.p_as_2d() / const.p_0())
            for key in const_p_wells:
                s_o_hist[key].append(s_o[key[0], key[1]])
                s_w_hist[key].append(s_w[key[0], key[1]])

            for key in neibours:
                s_o_hist[key].append(s_o[key[0], key[1]])
                s_w_hist[key].append(s_w[key[0], key[1]])

            plot_stuff_iter(q_w_rate=q_w_rate, q_o_rate=q_o_rate, const_p_wells=const_p_wells, s_o_hist=s_o_hist,
                            s_w_hist=s_w_hist, s_o_matrix_history=s_o_matrix_history, neibours=neibours,
                            p_matrix_history=p_matrix_history
                            )
