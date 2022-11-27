"""
Helper functions for visualizations
"""

from matplotlib import pyplot as plt
from numpy.typing import ArrayLike
import numpy as np
from matplotlib import patches


def setup_pyplot_params():
    """
    Sets pyplot params
    """
    plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif"})


def plot_hyperplane(
    xspace: ArrayLike,
    intercept: float,
    xslope: float,
    yslope: float,
    n: int = 3,
    ax: plt.Axes = None,
    normalize_arrows: bool = True,
    c: str = None,
    alpha: float = None,
    plot_kwargs: dict = None,
    quiver_kwargs: dict = None,
    return_artists: bool = False,
):
    """
    Visalize a 3D plane as a 2D line with arrows signaling which direction the 3D hyperplane is increasing

    Parameters
    ----------
    xspace : ArrayLikw
        Visualization domain

    intercept : float
        y-intercept of the 3D plane

    xslope : float
        slope of the 3D plane in the x-axis

    yslope : float
        slope of the 3D plane in the y-axis

    n : int
        Number of arrows to show

    ax : plt.Axes, optional
        axes to plot in

    normalize_arrows : bool, default = True
        normalize arrow sizes

    c : str, optional
        color of line and arrows

    alpha : str, optional
        alpha of line and arrows

    plot_kwargs : dict, optional
        kwargs to send to plt.plot

    quiver_kwargs : dict, optional
        kwargs to send to plt.quiver

    return_artists : bool, default=False
        Whether to return the lines and arrows or not

    Returns
    -------
    plt.Axes or tuple plt.Axes and dict of artists
    """

    plot_kwargs = plot_kwargs or {}
    quiver_kwargs = quiver_kwargs or {}
    if c is not None:
        plot_kwargs["c"] = c
        quiver_kwargs["color"] = c

    if alpha is not None:
        plot_kwargs["alpha"] = alpha
        quiver_kwargs["alpha"] = alpha

    ax = ax or plt.gca()
    xspace = np.array(xspace)

    a, b = -xslope / yslope, -intercept / yslope

    def f(x):
        return a * x + b

    line = ax.plot(xspace, f(xspace), **plot_kwargs)

    xmin, xmax = xspace.min(), xspace.max()
    diff = (xmax - xmin) / (n + 1)
    arrowxs = np.linspace(xmin + diff, xmax - diff, n)

    norm = 1
    if normalize_arrows:
        norm = np.linalg.norm([xslope, yslope])

    arrows = ax.quiver(arrowxs, f(arrowxs), xslope / norm, yslope / norm, **quiver_kwargs)
    if return_artists:
        return ax, {"line": line, "arrows": arrows}
    else:
        return ax


def _get_centered_points(x: float, n: int, spacing: float) -> np.ndarray:
    """
    Given a center x, generate n points centered around x

    Parameters
    ----------
    x : float
        The center

    n : int
        Number of points

    spacing : float
        Spacing of the points
    """
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
    edges: bool = True
):
    """
    Visualization of a dense artificial neural networks as a directed graph

    Parameters
    ----------
    layers : list of ints
        Nodes in each layer

    spacing : tuple of floats
        horizontal and vertical spacing of nodes respectively

    radius : float
        Radius of nodes

    ax : plt.Axes, optional
        axes to draw in

    circle_kwargs : dict, optional
        kwargs for patches.Circle (used to draw nodes)

    quiver_kwargs : dict, optional
        kwargs for the directed graph edges

    edges: bool, default=True
        To draw edges or not

    Returns
    -------
    list of lists of pyplot circle patches
    """

    ax = ax or plt.gca()
    circle_kwargs = {} if circle_kwargs is None else circle_kwargs
    quiver_kwargs = {} if quiver_kwargs is None else quiver_kwargs

    spacing = radius + np.array(spacing)  # Convert to node center spacing
    n = len(layers)

    circles = []
    for x, width in zip(_get_centered_points(center[0], n, spacing[0]), layers):
        circles.append([])
        for y in _get_centered_points(center[1], width, spacing[1])[::-1]:
            circle = patches.Circle((x, y), radius, **circle_kwargs)
            ax.add_patch(circle)
            circles[-1].append(circle)

    if not edges:
        return circles

    for left_circles, right_circles in zip(circles, circles[1:]):
        left_centers = np.array([l.get_center() for l in left_circles])
        right_centers = np.array([r.get_center() for r in right_circles])
        pdiff = (right_centers - left_centers[:, None, :]).reshape(-1, 2)
        adjust = pdiff / np.linalg.norm(pdiff, axis=1).reshape(-1, 1) * radius
        left_centers = left_centers.repeat(len(right_centers), 0) + adjust
        pdiff -= adjust * 2
        plt.quiver(*left_centers.T, *pdiff.T, units="xy", scale=1, scale_units="xy", **quiver_kwargs)

    return circles


if __name__ == "__main__":

    draw_ann(
        [2, 2, 3],
        center=(0, 0),
        spacing=(3, 2),
        circle_kwargs={"facecolor": (0, 0, 0, 0), "edgecolor": "k"},
        quiver_kwargs={"color": "k", "width": 0.1},
    )

    plt.gca().set_aspect("equal")
    plt.autoscale()
    plt.show()
