import numpy as np


def corner(x, y):
    out = (1 - np.sqrt(x ** 2 + y ** 2)) / 2
    out += ((x - 1) ** 2 + (y - 1) ** 2) / 8
    out = np.exp(out)
    return out


def corner_1(x, y):
    out = (1 - np.sqrt(x ** 2 + y ** 2)) / 2
    out += ((x + 1) ** 2 + (y + 1) ** 2) / 8
    out = np.exp(out)
    return out


def centroid(x, y):
    out = x ** 2 + y ** 2
    out = np.exp(- out / 2)
    return out


def get_matrix_with_pdf(nx: int, ny: int, pdf):
    out = np.zeros((nx, ny))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = pdf(x=(i - out.shape[0] / 3.) / out.shape[0],
                            y=(j - out.shape[1] / 2.) / out.shape[0])
    out /= out.max()
    return out
