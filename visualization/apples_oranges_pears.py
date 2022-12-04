"""
To create figures and animations that are related to apple, orange and pear classes
"""

from pprint import pprint
import sys
import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import patches
from matplotlib.colors import to_rgb
import math
from tqdm import tqdm
from pathlib import Path
import logging

project_dir = Path(__file__).resolve().parent.parent
figures_dir = project_dir / "visualization" / "figures"
sys.path.insert(0, str(project_dir))

from utils import get_lims, plot_hyperplane, unnormalize_planes, draw_ann, setup_pyplot_params, json_to_weights, plot_kwargs, quiver_kwargs

logger = logging.getLogger("visualize.apples_oranges_pears")

setup_pyplot_params()

try:
    df = pd.read_csv(project_dir / "data" / "generated" / "apples_oranges_pears.csv")
except Exception as e:
    logger.error(f"Something wrong when attempting to import data: {e}")
    sys.exit()

X, y = df[["weight", "height"]].values, df["class"].map({"apple": 0, "orange": 1, "pear": 2}).values
x_lim, y_lim = get_lims(X, padding=[0.75, 1.5])

m = X.mean(0)
s = X.std(0)

model_weights_2lp = json_to_weights(project_dir / "models" / "weights" / "2LP.json")
u2lp_biases, u2lp_weights = unnormalize_planes(m, s, model_weights_2lp["planes.bias"], model_weights_2lp["planes.weight"])

model_weights_3lp = json_to_weights(project_dir / "models" / "weights" / "3LP.json")
uhidden_biases, uhidden_weights = unnormalize_planes(m, s, model_weights_3lp["hidden.bias"], model_weights_3lp["hidden.weight"])
output_biases, output_weights = model_weights_3lp["output.bias"], model_weights_3lp["output.weight"]


def forward_sigmoid(X: np.ndarray, bias: np.ndarray, weights: np.ndarray) -> np.ndarray:
    z = bias + X @ weights.T
    return 1 / (1 + np.exp(-z))


h1 = forward_sigmoid(X, uhidden_biases, uhidden_weights)

xspace = torch.linspace(x_lim[0], x_lim[1], 4)

quiver_kwargs_animation = {**quiver_kwargs, "width": 2, "headwidth": 8, "scale": 0.05}
classes = np.array(["Apple", "Orange", "Pear"])
colors = np.array(
    [
        to_rgb("greenyellow"),
        to_rgb("orange"),
        to_rgb("forestgreen"),
    ]
)


def savefig(file: str | Path):
    file = figures_dir / file
    plt.savefig(file)
    logger.info(f"Created figure {file}")


def viz_decorator(file: str | Path):
    file = Path(file)

    def decorator(f):
        def wrapper(save: bool = True, clf: bool = True, **kwargs):
            return_value = f(**kwargs)
            if save:
                if file.suffix in {".gif", ".mp4"}:
                    # Function gotta return the animation object
                    anim, defer = return_value
                    anim.save(figures_dir / file, writer="ffmpeg", fps=60)
                    defer()
                    logger.info(f"Created animation {file}")
                else:
                    savefig(file)

            if clf:
                plt.clf()

            return return_value

        return wrapper

    return decorator


@viz_decorator("dataset_apples_oranges_pears.pdf")
def visualize_data_set(scatter_kwargs: dict = None, ax: plt.Axes = None):
    scatter_kwargs = scatter_kwargs or {}

    if ax is None:
        ax = plt.gca()
        plt.gcf().set_figheight(5)
        plt.gcf().set_figwidth(10)

    ax.set_xlabel("Weight (g)")
    ax.set_ylabel("Diameter (cm)")
    ax.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black", **scatter_kwargs)
    ax.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black", **scatter_kwargs)
    ax.scatter(*X[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20, **scatter_kwargs)
    ax.legend(loc="upper right")
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_aspect("equal")

    plt.title("Comparing apples, oranges and pears")
    # plt.show()
    return ax


@viz_decorator("apples_oranges_pears_with_apple_line.pdf")
def visualize_data_with_apple_line():
    ax = visualize_data_set(False, False)
    plt.title("Comparing apples, oranges and pears with a single decision boundary")

    plot_hyperplane(
        xspace,
        uhidden_biases[0],
        *uhidden_weights[0],
        10,
        ax=ax,
        c="k",
        plot_kwargs={**plot_kwargs, "label": "Decision boundary"},
        quiver_kwargs=quiver_kwargs,
    )

    ax.get_legend().remove()
    ax.legend(loc="upper right")
    # plt.show()


@viz_decorator("apples_oranges_pears_with_hidden_lines.pdf")
def visualize_data_with_hidden_lines(ax: plt.Axes = None, scatter_kwargs: dict = None, quiver_kwargs_: dict = None):
    if ax is None:
        ax = plt.gca()

    if scatter_kwargs is None:
        scatter_kwargs = {}

    if quiver_kwargs_ is None:
        quiver_kwargs_ = quiver_kwargs

    visualize_data_set(False, False, scatter_kwargs=scatter_kwargs, ax=ax)
    plt.title("Decision boundaries for apples and pears")

    artists = {}
    artists["lines"] = lines = []
    for (class_, color, bias, weights) in zip(classes, colors[[0, 2]], uhidden_biases, uhidden_weights):
        _, artists_ = plot_hyperplane(
            xspace,
            bias,
            *weights,
            10,
            c=color,
            plot_kwargs={**plot_kwargs, "label": f"{class_} line"},
            quiver_kwargs=quiver_kwargs_,
            ax=ax,
            return_artists=True,
        )
        lines.append(artists_["line"][0])

    ax.legend(loc="upper right")
    # plt.show()
    return artists


@viz_decorator("apples_oranges_pears_2lp_lines.pdf")
def visualize_data_with_2lp_lines():
    visualize_data_set(False, False)
    plt.title("Decision boundaries for apples, oranges and pears")

    for (class_, color, bias, weights) in zip(classes, colors, u2lp_biases, u2lp_weights):
        plot_hyperplane(
            xspace,
            bias,
            *weights,
            10,
            c=color,
            quiver_kwargs=quiver_kwargs,
            plot_kwargs={**plot_kwargs, "label": f"{class_} line"},
        )

    plt.legend()


circle_kwargs = {"facecolor": (0, 0, 0, 0), "edgecolor": "k"}
unknown_point_kwargs = {"marker": "x", "c": "black", "s": 70}


def calc_linewidth(x: float) -> float:
    """
    x is assumed to be between 0 and 1
    """
    return max(x * 18, plot_kwargs["linewidth"])


@viz_decorator("apples_oranges_pears_2lp_activations.pdf")
def visualize_2lp_activations(point: np.ndarray = None, axes: tuple[plt.Axes, plt.Axes] = None, quiver_kwargs_: dict = None):
    if quiver_kwargs_ is None:
        quiver_kwargs_ = quiver_kwargs

    if axes is None:
        fig, (ax_upper, ax_lower) = plt.subplots(2, 1, figsize=(10, 7))
        fig.tight_layout()
    else:
        ax_upper, ax_lower = axes

    if point is None:
        point = np.array([[140, 10]])

    visualize_data_set(False, False, scatter_kwargs={"alpha": 0.5}, ax=ax_upper)

    artists = {}
    artists["scatter"] = ax_upper.scatter(*point.T, label="Unknown", **unknown_point_kwargs, zorder=100)
    activations = forward_sigmoid(point, u2lp_biases, u2lp_weights)[0]

    # Lines
    artists["lines"] = lines = []
    for i, (class_, color) in enumerate(zip(classes, colors)):
        _, artists_ = plot_hyperplane(
            xspace,
            u2lp_biases[i],
            *u2lp_weights[i],
            8,
            c=color,
            plot_kwargs={
                **plot_kwargs,
                "linewidth": calc_linewidth(activations[i]),
                "label": class_ + " boundary",
            },
            quiver_kwargs=quiver_kwargs_,
            ax=ax_upper,
            return_artists=True,
        )
        lines.append(artists_["line"][0])

    # Model graph
    radius = 0.25
    ann_center = (-1.3, -0.1)
    fontsize = 13
    circles = draw_ann(
        layers=[2, 3],
        radius=radius,
        center=ann_center,
        spacing=(0.5, 0.3),
        ax=ax_lower,
        circle_kwargs=circle_kwargs,
        quiver_kwargs={"color": "k", "width": 0.016},
    )
    ccenters = [[np.array(circle.get_center()) for circle in layer] for layer in circles]

    # Output node colors
    circles[1][0].set_facecolor((*colors[0], activations[0]))
    circles[1][1].set_facecolor((*colors[1], activations[1]))
    circles[1][2].set_facecolor((*colors[2], activations[2]))

    artists["out0_circle"] = circles[1][0]
    artists["out1_circle"] = circles[1][1]
    artists["out2_circle"] = circles[1][2]

    # Static text in ax_lower
    ax_lower.annotate("Weight", ccenters[0][0] - [radius + 0.1, 0], ha="right", va="center", fontsize=fontsize)
    ax_lower.annotate("Diameter", ccenters[0][1] - [radius + 0.1, 0], ha="right", va="center", fontsize=fontsize)
    ax_lower.annotate("Model graph", ccenters[0][0] + [0.4, 0.7], ha="center", va="center", fontsize=14)
    ax_lower.annotate("Appleness", ccenters[1][0] + [radius + 0.1, 0], ha="left", va="center", fontsize=fontsize)
    ax_lower.annotate("Orangeness", ccenters[1][1] + [radius + 0.1, 0], ha="left", va="center", fontsize=fontsize)
    ax_lower.annotate("Pearness", ccenters[1][2] + [radius + 0.1, 0], ha="left", va="center", fontsize=fontsize)

    # Text inside nodes
    artists["x_text"] = ax_lower.annotate(f"{point[0, 0]:.1f}", ccenters[0][0], ha="center", va="center", fontsize=fontsize)
    artists["y_text"] = ax_lower.annotate(f"{point[0, 1]:.1f}", ccenters[0][1], ha="center", va="center", fontsize=fontsize)
    artists["out0_text"] = ax_lower.annotate(f"{activations[0]:.2f}", ccenters[1][0], ha="center", va="center", fontsize=fontsize)
    artists["out1_text"] = ax_lower.annotate(f"{activations[1]:.2f}", ccenters[1][1], ha="center", va="center", fontsize=fontsize)
    artists["out2_text"] = ax_lower.annotate(f"{activations[2]:.2f}", ccenters[1][2], ha="center", va="center", fontsize=fontsize)

    # Current class node
    curr_class = np.argmax(activations)
    class_ccenter = np.array([1.2, 0])
    class_cradius = radius * 1.75
    ax_lower.annotate(
        "Current classification",
        class_ccenter + [0, class_cradius + 0.2],
        ha="center",
        va="center",
        fontsize=14,
    )
    artists["class_circle"] = class_circle = patches.Circle(class_ccenter, radius=class_cradius, **circle_kwargs)
    class_circle.set_facecolor(colors[curr_class])
    ax_lower.add_patch(class_circle)

    artists["class_text"] = ax_lower.annotate(classes[curr_class], class_ccenter, ha="center", va="center", fontsize=14)

    ax_upper.legend(loc="lower right")
    ax_upper.set_title("")

    ax_lower.set_aspect("equal")
    ax_lower.set_xlim(-3, 3)
    ax_lower.set_ylim(-1, 1)
    ax_lower.set_title("")
    ax_lower.set_xticks([])
    ax_lower.set_yticks([])
    ax_lower.axis("off")

    fig.suptitle("Activations")

    # plt.show()

    return fig, (ax_upper, ax_lower), artists


@viz_decorator("2lp_activations.gif")
def visualize_2lp_activations_animated():
    fig, (ax_upper, ax_lower), artists = visualize_2lp_activations(False, False, quiver_kwargs_=quiver_kwargs_animation)

    logger.info("Rendering 2LP animation")
    n = 300  # Animation steps
    pi2 = np.pi * 2
    pbar = tqdm(total=n + 1, disable=False)
    artists_to_animate = list(pd.core.common.flatten(artists.values()))

    def step(i: int):
        rad = pi2 * i / n
        point = (m + (np.cos(rad) * 20, np.sin(rad) * 5))[None]
        artists["scatter"].set_offsets(point)
        activations = forward_sigmoid(point, u2lp_biases, u2lp_weights)[0]

        for i in range(3):
            artists[f"out{i}_text"].set_text(f"{activations[i]:.2f}")
            artists[f"out{i}_circle"].set_facecolor((*colors[i], activations[i]))
            artists["lines"][i].set_linewidth(calc_linewidth(activations[i]))

        artists["x_text"].set_text(f"{point[0, 0]:.1f}")
        artists["y_text"].set_text(f"{point[0, 1]:.1f}")

        curr_class = np.argmax(activations)
        artists["class_text"].set_text(classes[curr_class])
        artists["class_circle"].set_facecolor(colors[curr_class])

        pbar.update(1)
        return artists_to_animate

    ax_upper.get_legend().remove()
    anim = FuncAnimation(fig, step, blit=True, interval=0, frames=n)
    # plt.show()
    return anim, lambda: pbar.close()


@viz_decorator("appleness_pearness.pdf")
def visualize_appleness_pearness(axes: tuple[plt.Axes, plt.Axes] = None, scatter_kwargs: dict = None, quiver_kwargs_: dict = None):
    if axes is None:
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(10, 5.5))
    else:
        fig = plt.gcf()
        ax_left, ax_right = axes

    if scatter_kwargs is None:
        scatter_kwargs = {}

    if quiver_kwargs_ is None:
        quiver_kwargs_ = quiver_kwargs

    artists = visualize_data_with_hidden_lines(False, False, ax=ax_left, scatter_kwargs=scatter_kwargs, quiver_kwargs_=quiver_kwargs_)

    ax_left.set_xlabel("Weight (g)")
    ax_left.set_ylabel("Diameter (cm)")
    ax_left.set_aspect("equal")
    ax_left.set_title("Lines for apples and pears")
    ax_left.set_ylim(*y_lim)
    ax_left.get_legend().remove()

    ax_right.scatter(*h1[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black", **scatter_kwargs)
    ax_right.scatter(*h1[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black", **scatter_kwargs)
    ax_right.scatter(*h1[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20, **scatter_kwargs)
    ax_right.set_xlabel("Appleness")
    ax_right.set_ylabel("Pearness")
    ax_right.set_title("Activation space")
    ax_right.set_aspect("equal")
    ax_right.legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.suptitle("Visualizing appleness and pearness for each point")
    # plt.show()
    return fig, (ax_left, ax_right), artists


@viz_decorator("appleness_pearness_with_out_lines.pdf")
def visualize_appleness_pearness_out_lines(
    axes: tuple[plt.Axes, plt.Axes] = None, scatter_kwargs: dict = None, quiver_kwargs_: dict = None
):

    if scatter_kwargs is None:
        scatter_kwargs = {}

    if quiver_kwargs_ is None:
        quiver_kwargs_ = quiver_kwargs

    if axes is None:
        fig, (ax_left, ax_right), artists = visualize_appleness_pearness(
            False, False, scatter_kwargs=scatter_kwargs, quiver_kwargs_=quiver_kwargs_
        )
    else:
        ax_left, ax_right = axes
        fig, _, artists = visualize_appleness_pearness(
            False, False, axes=axes, scatter_kwargs=scatter_kwargs, quiver_kwargs_=quiver_kwargs_
        )

    artists["hidden_lines"] = artists.pop("lines")

    h1lims = get_lims(h1)
    xspace = np.linspace(*h1lims[0], 4)
    artists["out_lines"] = lines = []
    for i, (class_, color) in enumerate(zip(classes, colors)):
        _, artists_ = plot_hyperplane(
            xspace,
            output_biases[i],
            *output_weights[i],
            6,
            c=color,
            ax=ax_right,
            quiver_kwargs=quiver_kwargs_,
            plot_kwargs={**plot_kwargs, "label": f"{class_} line"},
            return_artists=True,
        )
        lines.append(artists_["line"][0])

    ax_right.set_xlim(*h1lims[0])
    ax_right.set_ylim(*h1lims[1])
    ax_right.legend(loc="upper right")

    # plt.show()
    return fig, (ax_left, ax_right), artists


@viz_decorator("3lp.pdf")
def visualize_3lp(point: np.ndarray = None):
    fig = plt.figure(figsize=(10, 8))
    ax_upperleft = fig.add_subplot(221)
    ax_upperright = fig.add_subplot(222)
    ax_bottom = fig.add_subplot(212)

    _, _, artists = visualize_appleness_pearness_out_lines(
        False, False, axes=(ax_upperleft, ax_upperright), scatter_kwargs={"alpha": 0.25}, quiver_kwargs_=quiver_kwargs_animation
    )
    # Artist keys: hidden_lines, out_lines

    point0 = np.array([point or [140, 10]], dtype=float)
    h1 = forward_sigmoid(point0, uhidden_biases, uhidden_weights)[0]
    point1 = h1
    h2 = forward_sigmoid(point1, output_biases, output_weights)

    # "Unknown" scatter points
    artists["point0"] = ax_upperleft.scatter(*point0[None].T, label="Unknown", **unknown_point_kwargs, zorder=100)
    artists["point1"] = ax_upperright.scatter(*point1[None].T, label="Unknown", **unknown_point_kwargs, zorder=100)

    # Lines with varying widths
    for i, hidden_line in enumerate(artists["hidden_lines"]):
        hidden_line.set_linewidth(calc_linewidth(h1[i]))

    for i, out_line in enumerate(artists["out_lines"]):
        out_line.set_linewidth(calc_linewidth(h2[i]))

    # Model graph
    radius = 0.25
    ann_center = (-1.3, 0)
    fontsize = 13
    circle_kwargs = {"facecolor": (0, 0, 0, 0), "edgecolor": "k"}
    circles = draw_ann(
        layers=[2, 2, 3],
        radius=radius,
        center=ann_center,
        spacing=(0.5, 0.4),
        ax=ax_bottom,
        circle_kwargs=circle_kwargs,
        quiver_kwargs={"color": "k", "width": 0.016},
    )
    artists["circles"] = circles[1:]
    ccenters = [[np.array(circle.get_center()) for circle in layer] for layer in circles]

    # Hidden node colors
    circles[1][0].set_facecolor(colors[0])
    circles[1][1].set_facecolor(colors[2])

    # Output node colors
    circles[2][0].set_facecolor(colors[0])
    circles[2][1].set_facecolor(colors[1])
    circles[2][2].set_facecolor(colors[2])

    # Static text in ax_bottom
    ax_bottom.annotate("Weight", ccenters[0][0] - [radius + 0.1, 0], ha="right", va="center", fontsize=fontsize)
    ax_bottom.annotate("Diameter", ccenters[0][1] - [radius + 0.1, 0], ha="right", va="center", fontsize=fontsize)
    ax_bottom.annotate("Model graph", ccenters[1][0] + [0, 0.6], ha="center", va="center", fontsize=14)
    ax_bottom.annotate("Appleness", ccenters[2][0] + [radius + 0.1, 0], ha="left", va="center", fontsize=fontsize)
    ax_bottom.annotate("Orangeness", ccenters[2][1] + [radius + 0.1, 0], ha="left", va="center", fontsize=fontsize)
    ax_bottom.annotate("Pearness", ccenters[2][2] + [radius + 0.1, 0], ha="left", va="center", fontsize=fontsize)

    # Text inside nodes
    artists["x_text"] = x_text = ax_bottom.annotate("x", ccenters[0][0], ha="center", va="center", fontsize=fontsize)
    artists["y_text"] = y_text = ax_bottom.annotate("y", ccenters[0][1], ha="center", va="center", fontsize=fontsize)
    artists["hidden1_text"] = hidden1_text = ax_bottom.annotate("h1", ccenters[1][0], ha="center", va="center", fontsize=fontsize)
    artists["hidden2_text"] = hidden2_text = ax_bottom.annotate("h2", ccenters[1][1], ha="center", va="center", fontsize=fontsize)
    artists["out1_text"] = out1_text = ax_bottom.annotate("o1", ccenters[2][0], ha="center", va="center", fontsize=fontsize)
    artists["out2_text"] = out2_text = ax_bottom.annotate("o2", ccenters[2][1], ha="center", va="center", fontsize=fontsize)
    artists["out3_text"] = out3_text = ax_bottom.annotate("o3", ccenters[2][2], ha="center", va="center", fontsize=fontsize)

    x_text.set_text(f"{point0[0, 0]:.1f}")
    y_text.set_text(f"{point0[0, 1]:.1f}")

    # Current class node
    class_ccenter = np.array([1.6, 0])
    class_cradius = radius * 1.75
    ax_bottom.annotate(
        "Current classification",
        class_ccenter + [0, class_cradius + 0.2],
        ha="center",
        va="center",
        fontsize=14,
    )
    artists["class_circle"] = class_circle = patches.Circle(class_ccenter, radius=class_cradius, **circle_kwargs)
    artists["class_text"] = class_text = ax_bottom.annotate("Apple", class_ccenter, ha="center", va="center", fontsize=14)
    ax_bottom.add_patch(class_circle)

    hidden1_text.set_text(f"{h1[0]:.2f}")
    hidden2_text.set_text(f"{h1[1]:.2f}")
    circles[1][0].set_facecolor((*circles[1][0].get_facecolor()[:3], h1[0]))
    circles[1][1].set_facecolor((*circles[1][1].get_facecolor()[:3], h1[1]))

    out1_text.set_text(f"{h2[0]:.2f}")
    out2_text.set_text(f"{h2[1]:.2f}")
    out3_text.set_text(f"{h2[2]:.2f}")
    circles[2][0].set_facecolor((*circles[2][0].get_facecolor()[:3], h2[0]))
    circles[2][1].set_facecolor((*circles[2][1].get_facecolor()[:3], h2[1]))
    circles[2][2].set_facecolor((*circles[2][2].get_facecolor()[:3], h2[2]))

    curr_class = np.argmax(h2)
    class_text.set_text(classes[curr_class])
    class_circle.set_facecolor(colors[curr_class])

    ax_upperright.get_legend().remove()

    ax_bottom.set_title("")
    ax_bottom.set_aspect("equal")
    ax_bottom.set_xlim(-3, 3)
    ax_bottom.set_ylim(-1, 1)
    ax_bottom.set_xticks([])
    ax_bottom.set_yticks([])
    ax_bottom.axis("off")

    fig.tight_layout()

    # plt.show()
    return fig, (ax_upperleft, ax_upperright, ax_bottom), artists


@viz_decorator("3lp.gif")
def visualize_3lp_animated():
    fig, (ax_upperleft, ax_upperright, ax_bottom), artists = visualize_3lp(False, False)

    hidden_lines = artists["hidden_lines"]
    out_lines = artists["out_lines"]
    scatterpoint0 = artists["point0"]
    scatterpoint1 = artists["point1"]
    circles = artists["circles"]
    x_text = artists["x_text"]
    y_text = artists["y_text"]
    hidden1_text = artists["hidden1_text"]
    hidden2_text = artists["hidden2_text"]
    out1_text = artists["out1_text"]
    out2_text = artists["out2_text"]
    out3_text = artists["out3_text"]
    class_text = artists["class_text"]
    class_circle = artists["class_circle"]

    logger.info("Rendering 3LP animation")

    n = 300
    pbar = tqdm(total=n + 1, disable=False)
    centerx = np.mean(x_lim)
    centery = np.mean(y_lim)

    point0 = np.array([[0, 0]], dtype=float)

    artists_to_animate = list(pd.core.common.flatten(artists.values()))
    pi2 = np.pi * 2

    def step(i):
        rad = i / n * pi2
        point0[0, 0] = centerx + 20 * math.cos(rad)
        point0[0, 1] = centery + 5 * math.sin(rad)

        h1 = forward_sigmoid(point0, uhidden_biases, uhidden_weights)[0]
        point1 = h1
        h2 = forward_sigmoid(point1, output_biases, output_weights)

        scatterpoint0.set_offsets(point0)
        scatterpoint1.set_offsets(point1)

        # Lines with varying widths
        for i, hidden_line in enumerate(hidden_lines):
            hidden_line.set_linewidth(calc_linewidth(h1[i]))

        for i, out_line in enumerate(out_lines):
            out_line.set_linewidth(calc_linewidth(h2[i]))

        # Hidden node colors
        circles[0][0].set_facecolor((*colors[0], h1[0]))
        circles[0][1].set_facecolor((*colors[2], h1[1]))

        # Output node colors
        circles[1][0].set_facecolor((*colors[0], h2[0]))
        circles[1][1].set_facecolor((*colors[1], h2[1]))
        circles[1][2].set_facecolor((*colors[2], h2[2]))

        # Text stuff
        x_text.set_text(f"{point0[0, 0]:.1f}")
        y_text.set_text(f"{point0[0, 1]:.1f}")
        hidden1_text.set_text(f"{h1[0]:.2f}")
        hidden2_text.set_text(f"{h1[1]:.2f}")
        out1_text.set_text(f"{h2[0]:.2f}")
        out2_text.set_text(f"{h2[1]:.2f}")
        out3_text.set_text(f"{h2[2]:.2f}")

        # Class circle
        curr_class = np.argmax(h2)
        class_text.set_text(classes[curr_class])
        class_circle.set_facecolor(colors[curr_class])

        pbar.update(1)
        return artists_to_animate

    anim = FuncAnimation(fig, step, blit=True, interval=0, frames=n)

    # plt.show

    return anim, lambda: pbar.close()


if __name__ == "__main__":

    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    visualize_data_set()
    visualize_data_with_apple_line()
    visualize_data_with_hidden_lines()
    visualize_data_with_2lp_lines()
    visualize_2lp_activations()
    visualize_2lp_activations_animated()
    visualize_appleness_pearness()
    visualize_appleness_pearness_out_lines()
    visualize_3lp()
    visualize_3lp_animated()
