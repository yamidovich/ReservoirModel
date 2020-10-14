from sklearn.datasets import make_moons, make_circles
import pandas as pd
import numpy as np
from scipy import ndimage


def get_saturation_scattered(n_samples=500, noise=0.4, random_state=None) -> tuple:
    """
    returns scatter points for interpretation as water and oil
    shape - moons
    :param n_samples: number of samples
    :param noise: how water and oil are mixed
    :param random_state: rand state
    :return: (x, y). x - np.ndarray(n_samples, 2) for 2 coordinates. y - {0, 1} oil or water
    """
    return make_moons(n_samples=n_samples, noise=noise, random_state=random_state)


def get_depth_scattered(n_samples=500, noise=0.3, random_state=None, factor=0.4):
    ds = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    y = ds[1] * 1.5 - 0.5 * np.ones(ds[1].shape)
    return ds[0], y


def get_porosity_scattered(n_samples=500, noise=0.4, random_state=None, factor=0.4):
    ds = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    y = ds[1] * 1.5 - 0.5 * np.ones(ds[1].shape)
    return ds[0], y


def get_grid_from_scattered(ds: pd.DataFrame,
                            y_min: float = -1.5, y_max: float = 1.5,
                            x_min: float = -1.5, x_max: float = 1.5,
                            n_x: int = 25, n_y: int = 25) -> np.ndarray:
    """
    from scattered df with point returns a greed as a matrix.
    :param ds: (np.ndarray(2, n), np.ndarray(n) in {1, 2}^n )
    :param y_min: as distribution for scattered
    bound may be generated randomly here is a handcrafted bound
    :param y_max: as distribution for scattered
    bound may be generated randomly here is a handcrafted bound
    :param x_min: as distribution for scattered
    bound may be generated randomly here is a handcrafted bound
    :param x_max: as distribution for scattered
    bound may be generated randomly here is a handcrafted bound
    :param n_x: number of cells in x component
    :param n_y: number of cells in x component
    :return: np.ndarray(n_y, n_xx) in [0,1]^{n_x * n_y} or some cells can be equal
     to -1 as cells with no samples
    """
    xs = np.linspace(x_min, x_max, n_x + 1)
    ys = np.linspace(y_min, y_max, n_y + 1)
    greed_matrix = np.nan * np.ones((n_x, n_y))

    for ix in range(len(xs) - 1):
        for iy in range(len(ys) - 1):
            x_filter = (ds[0][:, 0] > xs[ix]) & (ds[0][:, 0] < xs[ix + 1])
            y_filter = (ds[0][:, 1] > ys[iy]) & (ds[0][:, 1] < ys[iy + 1])
            _filter = (y_filter & x_filter)
            if _filter.sum() > 0:
                greed_matrix[ix][iy] = ds[1][_filter].mean() * 0.5 + 0.5
    greed_matrix = ndimage.generic_filter(greed_matrix, np.nanmean, size=3, mode='constant', cval=np.NaN) * 0.5 + 0.5
    return greed_matrix
