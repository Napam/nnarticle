from numbers import Number
import numpy as np
from numpy.typing import ArrayLike


def get_lims(X: ArrayLike, padding: float | ArrayLike = 0.25) -> tuple[tuple[float, float], tuple[float, float]]:
    """
    Given a 2D dataset X, get x and y limits to be used with pyplot plots

    Parameters
    ----------
    X : ArrayLike
        A 2-dimensional dataset

    padding : float or ArrayLike of floats
        Adds more space around the lims

    Returns
    -------
    Tuple of tuple of floats
        That are (xlims, ylims)
    """
    if isinstance(padding, Number):
        padding = np.array([padding, padding])

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_std, y_std = X[:, 0].std(), X[:, 1].std()
    return (x_min - x_std * padding[0], x_max + x_std * padding[0]), (
        y_min - y_std * padding[1],
        y_max + y_std * padding[1],
    )


def normalize_data(X: ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize dataset X using standardization

    Parameters
    ----------
    X : ArrayLike
        Dataset

    Returns
    -------
    Tuple of numpy arrays
    Returns the normalized dataset, the feature means of the input dataset, the feature standard deviations of the input dataset
    """
    X_mean = X.mean(0)
    X_std = X.std(0)
    return (X - X_mean) / X_std, X_mean, X_std


def unnormalize_planes(m: ArrayLike, s: ArrayLike, biases: ArrayLike, weights: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """
    Given biases and weights of an input layer trained on normalized data (using standardization), get the re-adjusted biases and weights that will work on un-normalized data

    Parameters
    ----------
    m : ArrayLike
        The feature means of the un-normalized dataset

    s : ArrayLike
        The feature standard deviations of the un-normalized dataset

    biases : ArrayLike
        The biases of the input layer trained on normalized data

    weights : ArrayLike
        The weights of the input layer trained on normalized data

    Returns
    -------
    tuple of two numpy arrays
        The un-normalized biases and weights that can be used on unnormalized data
    """
    biases, weights = np.copy(biases), np.copy(weights)
    biases = biases - (m[0] * weights[:, 0]) / s[0] - (m[1] * weights[:, 1]) / s[1]
    weights = weights / s
    return biases, weights
