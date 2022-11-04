import torch
from matplotlib import pyplot as plt
from numbers import Number
from numpy.typing import ArrayLike
import numpy as np
from matplotlib import patches


def get_lims(X: torch.Tensor, padding: float | ArrayLike = 0.25):
    if isinstance(padding, Number):
        padding = np.array([padding, padding])

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    x_std, y_std = X[:, 0].std(), X[:, 1].std()
    return (x_min - x_std * padding[0], x_max + x_std * padding[0]), (y_min - y_std * padding[1], y_max + y_std * padding[1])


def plot_hyperplane(
        xspace: ArrayLike,
        intercept: float,
        xslope: float,
        yslope: float,
        n: int = 3,
        ax: plt.Axes = None,
        unit_plane: bool = True,
        c: str = None,
        alpha: float = None,
        plot_kwargs: dict = None,
        quiver_kwargs: dict = None,
        return_artists: bool = False,
):

    plot_kwargs = plot_kwargs or {}
    quiver_kwargs = quiver_kwargs or {}
    if c is not None:
        plot_kwargs['c'] = c
        quiver_kwargs['color'] = c

    if alpha is not None:
        plot_kwargs['alpha'] = alpha
        quiver_kwargs['alpha'] = alpha

    ax = ax or plt.gca()
    xspace = np.array(xspace)

    a, b = - xslope / yslope, - intercept / yslope
    def f(x): return a * x + b
    line = ax.plot(xspace, f(xspace), **plot_kwargs)

    xmin, xmax = xspace.min(), xspace.max()
    diff = (xmax - xmin) / (n + 1)
    arrowxs = np.linspace(xmin + diff, xmax - diff, n)

    norm = 1
    if unit_plane:
        norm = np.linalg.norm([xslope, yslope])

    arrows = ax.quiver(arrowxs, f(arrowxs), xslope / norm, yslope / norm, **quiver_kwargs)
    if return_artists:
        return ax, {'line': line, 'arrows': arrows}
    else:
        return ax


def normalize_data(X: ArrayLike):
    """Standardization"""
    X_mean = X.mean(0)
    X_std = X.std(0)
    return (X - X_mean) / X_std, X_mean, X_std


def unnormalize_plane(m: ArrayLike, s: ArrayLike, intercept: Number, xslope: Number, yslope: Number):
    intercept_ = intercept - (m[0] * xslope) / s[0] - (m[1] * yslope) / s[1]
    xslope_ = xslope / s[0]
    yslope_ = yslope / s[1]
    return intercept_, xslope_, yslope_


def unnormalize_planes(m: ArrayLike, s: ArrayLike, intercepts: ArrayLike, slopes: ArrayLike):
    intercepts, slopes = np.copy(intercepts), np.copy(slopes)
    intercepts = intercepts - (m[0] * slopes[:, 0]) / s[0] - (m[1] * slopes[:, 1]) / s[1]
    slopes = slopes / s
    return intercepts, slopes

def _get_centered_points(x: float, n: int, spacing: float) -> np.ndarray:
    length = (n - 1) * spacing
    start = x - length / 2
    stop = start + length
    return np.linspace(start, stop, n)

def draw_ann(
        layers: list[int],
        *,
        center: tuple[float] = (0, 0),
        spacing: tuple[float] = (3, 1.5),
        radius: float = 1,
        ax: plt.Axes = None,
        circle_kwargs: dict = None,
        quiver_kwargs: dict = None,
    ):

    ax = ax or plt.gca()
    circle_kwargs = {} if circle_kwargs is None else circle_kwargs
    quiver_kwargs = {} if quiver_kwargs is None else quiver_kwargs

    spacing = radius + np.array(spacing) # Convert to node center spacing
    n = len(layers)

    circles = []
    for i, x, width in zip(range(n), _get_centered_points(center[0], n, spacing[0]), layers):
        circles.append([])
        for j, y in enumerate(_get_centered_points(center[1], width, spacing[1])):
            circle = patches.Circle((x, y), radius, **circle_kwargs)
            ax.add_patch(circle)
            circles[-1].append(circle)

    for left_circles, right_circles in zip(circles, circles[1:]):
        left_centers = np.array([l.get_center() for l in left_circles])
        right_centers = np.array([r.get_center() for r in right_circles])
        pdiff = (right_centers - left_centers[:,None,:]).reshape(-1,2)
        adjust = pdiff / np.linalg.norm(pdiff, axis=1).reshape(-1,1) * radius
        left_centers = left_centers.repeat(len(right_centers), 0) + adjust
        pdiff -= adjust * 2
        plt.quiver(*left_centers.T, *pdiff.T, units="xy", scale=1, scale_units="xy", **quiver_kwargs)

    return circles

if __name__ == '__main__':

    # xspace = np.linspace(-2, 2, 10)
    # ax = plot_hyperplane(xspace, 0, 1, 1, n=10, c='r', quiver_kwargs={'units': 'dots'})
    # ax = plot_hyperplane(xspace, -1, 1, -1, n=10, c='g', quiver_kwargs={'units': 'dots'})
    # ax = plot_hyperplane(xspace, -1, -1, -1, n=10, c='b', quiver_kwargs={'units': 'dots'})
    # ax = plot_hyperplane(xspace, 0, -1, 1, n=10, c='y', quiver_kwargs={'units': 'dots'})
    # ax.set_aspect('equal')
    # ax.set_xlim((-10,10))
    # ax.set_ylim((-10,10))
    # plt.show()

    draw_ann(
        [2,2,3],
        center=(0,0),
        spacing=(3,2),
        circle_kwargs={"facecolor": (0,0,0,0), "edgecolor": "k"},
        quiver_kwargs={"color": "k", "width": 0.1}
    )
    plt.gca().set_aspect('equal')
    plt.autoscale()
    plt.show()

    pass
