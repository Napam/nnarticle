from numbers import Number
import numpy as np
from numpy.typing import ArrayLike


def get_lims(X: ArrayLike, padding: float | ArrayLike = 0.25):
    if isinstance(padding, Number):
        padding = np.array([padding, padding])

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_std, y_std = X[:, 0].std(), X[:, 1].std()
    return (x_min - x_std * padding[0], x_max + x_std * padding[0]), (
        y_min - y_std * padding[1],
        y_max + y_std * padding[1],
    )


def normalize_data(X: ArrayLike):
    """Standardization"""
    X_mean = X.mean(0)
    X_std = X.std(0)
    return (X - X_mean) / X_std, X_mean, X_std


def unnormalize_planes(m: ArrayLike, s: ArrayLike, intercepts: ArrayLike, slopes: ArrayLike):
    intercepts, slopes = np.copy(intercepts), np.copy(slopes)
    intercepts = intercepts - (m[0] * slopes[:, 0]) / s[0] - (m[1] * slopes[:, 1]) / s[1]
    slopes = slopes / s
    return intercepts, slopes
