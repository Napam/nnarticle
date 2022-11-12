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
from itertools import chain
from pathlib import Path
import logging

project_dir = Path(__file__).resolve().parent.parent
figures_dir = project_dir / "visualization" / "figures"
sys.path.insert(0, str(project_dir))

from utils import get_lims, plot_hyperplane, unnormalize_planes, draw_ann, setup_pyplot_params, json_to_weights

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
u2lp_biases, u2lp_weights = unnormalize_planes(
    m, s, model_weights_2lp["planes.bias"], model_weights_2lp["planes.weight"]
)

model_weights_3lp = json_to_weights(project_dir / "models" / "weights" / "3LP.json")
uhidden_biases, uhidden_weights = unnormalize_planes(
    m, s, model_weights_3lp["hidden.bias"], model_weights_3lp["hidden.weight"]
)
uoutput_biases, uoutput_weights = unnormalize_planes(
    m, s, model_weights_3lp["output.bias"], model_weights_3lp["output.weight"]
)

xspace = torch.linspace(x_lim[0], x_lim[1], 4)

plot_kwargs = {}
quiver_kwargs = {"units": "dots", "width": 2, "headwidth": 8, "scale": 0.05, "scale_units": "dots"}

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


def visualize_data_set(save: bool = True, clf: bool = True, scatter_kwargs: dict = None, ax: plt.Axes = None):
    scatter_kwargs = scatter_kwargs or {}

    if ax is None:
        ax = plt.gca()
        plt.gcf().set_figheight(10)
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

    if save:
        savefig("dataset_apples_oranges_pears.pdf")

    if clf:
        plt.clf()


def visualize_data_with_apple_line():
    visualize_data_set(False, False)
    plt.title("Comparing apples, oranges and pears with a single decision boundary")

    plot_hyperplane(xspace, uhidden_biases[0], *uhidden_weights[0], 10, c="k", quiver_kwargs=quiver_kwargs)

    savefig("apples_oranges_pears_with_apple_line.pdf")

    # plt.show()
    plt.clf()


def visualize_data_with_hidden_lines():
    visualize_data_set(False, False)
    plt.title("Decision boundaries for apples and pears")

    for (color, bias, weights) in zip(colors[[0, 2]], uhidden_biases, uhidden_weights):
        plot_hyperplane(xspace, bias, *weights, 10, c=color, quiver_kwargs=quiver_kwargs)

    savefig("apples_oranges_pears_hidden_lines.pdf")

    # plt.show()
    plt.clf()


def visualize_data_with_2lp_lines(save: bool = True, clf: bool = True):
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
            plot_kwargs={"linewidth": 4, "label": f"{class_} line"},
        )

    plt.legend()

    if save:
        savefig("apples_oranges_pears_2lp_lines.pdf")

    # plt.show()

    if clf:
        plt.clf()


def forward_sigmoid(X: np.ndarray, bias: np.ndarray, weights: np.ndarray) -> np.ndarray:
    z = bias + X @ weights.T
    return 1 / (1 + np.exp(-z))


circle_kwargs = {"facecolor": (0, 0, 0, 0), "edgecolor": "k"}
calc_linewidth = lambda x: max(x * 16, 2)


def visualize_2lp_activations(
    save: bool = True, clf: bool = True, point: np.ndarray = None, axes: tuple[plt.Axes, plt.Axes] = None
):
    if axes is None:
        fig, (ax_upper, ax_lower) = plt.subplots(2, 1, figsize=(10, 7))
        fig.tight_layout()
    else:
        ax_upper, ax_lower = axes

    if point is None:
        point = np.array([[140, 10]])

    visualize_data_set(False, False, {"alpha": 0.5}, ax_upper)

    artists = {}
    artists["scatter"] = ax_upper.scatter(*point.T, label="Unknown", marker="x", c="black", s=70)
    activations = forward_sigmoid(point, u2lp_biases, u2lp_weights)[0]

    # Lines
    artists["lines"] = lines = []
    artists["arrows"] = arrows = []
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
            quiver_kwargs=quiver_kwargs,
            ax=ax_upper,
            return_artists=True,
        )
        lines.append(artists_["line"][0])
        arrows.append(artists_["arrows"])

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
    ax_lower.annotate("Pearness", ccenters[1][1] + [radius + 0.1, 0], ha="left", va="center", fontsize=fontsize)
    ax_lower.annotate("Orangeness", ccenters[1][2] + [radius + 0.1, 0], ha="left", va="center", fontsize=fontsize)

    # Text inside nodes
    artists["x_text"] = ax_lower.annotate(
        f"{point[0, 0]:.1f}", ccenters[0][0], ha="center", va="center", fontsize=fontsize
    )
    artists["y_text"] = ax_lower.annotate(
        f"{point[0, 1]:.1f}", ccenters[0][1], ha="center", va="center", fontsize=fontsize
    )
    artists["out0_text"] = ax_lower.annotate(
        f"{activations[0]:.2f}", ccenters[1][0], ha="center", va="center", fontsize=fontsize
    )
    artists["out1_text"] = ax_lower.annotate(
        f"{activations[1]:.2f}", ccenters[1][1], ha="center", va="center", fontsize=fontsize
    )
    artists["out2_text"] = ax_lower.annotate(
        f"{activations[2]:.2f}", ccenters[1][2], ha="center", va="center", fontsize=fontsize
    )

    # Current class node
    curr_class = np.argmax(activations)
    class_ccenter = np.array([1.2, 0])
    class_cradius = radius * 1.75
    ax_lower.annotate(
        "Current classification", class_ccenter + [0, class_cradius + 0.2], ha="center", va="center", fontsize=14
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

    if save:
        savefig("apples_oranges_pears_activations.pdf")

    # plt.show()
    if clf:
        plt.clf()

    return fig, (ax_upper, ax_lower), artists


def visualize_2lp_activations_animated():
    fig, (ax_upper, ax_lower), artists = visualize_2lp_activations(False, False)

    logger.info("Rendering 2LP animation")
    n = 300  # Animation steps
    pi2 = np.pi * 2
    pbar = tqdm(total=n + 1, disable=False)
    artists_to_animate = pd.core.common.flatten(artists.values())
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
    file = figures_dir / "2lp.gif"
    anim.save(file, writer="ffmpeg", fps=60)
    logger.info(f"Saved animation at {file}")
    # plt.show()


def visualize_appleness_pearness():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    intercepts = np.array(
        [
            -1.9472151,  # Apple
            -2.260901,  # Pear
        ]
    )

    slopes = np.array(
        [
            [-4.1687274, -1.3713175],
            [4.5323997, -1.6058096],
        ]
    )

    m = np.array([141.8463, 6.2363])
    s = np.array([10.5088, 1.7896])

    xspace = np.linspace(x_lim[0], x_lim[1], 4)
    uintercepts, uslopes = unnormalize_planes(m, s, intercepts, slopes)

    plot_kwargs = {}
    quiver_kwargs = {"units": "dots", "width": 2, "headwidth": 4, "scale": 0.075, "scale_units": "dots"}

    classes = ["Apple", "Pear"]
    labels = ["Apple boundary", "Pear boundary"]
    colors = ["greenyellow", "forestgreen"]

    xspace = np.linspace(x_lim[0], x_lim[1], 4)
    ax1.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    ax1.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    ax1.scatter(*X[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20)
    for i, (label, color) in enumerate(zip(labels, colors)):
        _, artists = plot_hyperplane(
            xspace,
            uintercepts[i],
            *uslopes[i],
            5,
            c=color,
            plot_kwargs={**plot_kwargs, "label": label},
            quiver_kwargs=quiver_kwargs,
            return_artists=True,
            ax=ax1,
        )

    outs = uintercepts + X @ uslopes.T
    outs = 1 / (1 + np.exp(-outs))

    ax1.set_xlabel("Weight (g)")
    ax1.set_ylabel("Diameter (cm)")
    ax1.set_aspect("equal")
    ax1.set_title("Lines for apples and pears")
    ax1.set_ylim(*y_lim)

    ax2.scatter(*outs[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    ax2.scatter(*outs[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    ax2.scatter(*outs[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20)
    ax2.set_xlabel("Appleness")
    ax2.set_ylabel("Pearness")
    ax2.set_title("Activation space")
    ax2.set_aspect("equal")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.suptitle("Visualizing appleness and pearness for each point")
    plt.savefig("figures/appleness_pearness.pdf")
    # plt.show()
    plt.clf()


def visualize_appleness_pearness_out_lines():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    hidden_biases = np.array(
        [
            -1.9472151,  # Apple
            -2.260901,  # Pear
        ]
    )

    hidden_weights = np.array(
        [
            [-4.1687274, -1.3713175],
            [4.5323997, -1.6058096],
        ]
    )

    output_biases = np.array(
        [
            -2.0450604,  # Apple
            -2.1543744,  # Pear
            2.6014535,  # Orange
        ]
    )

    output_weights = np.array(
        [
            [5.4452653, -1.87916],
            [-2.0285792, 5.59163],
            [-4.693778, -5.0045652],
        ]
    )

    m = np.array([141.8463, 6.2363])
    s = np.array([10.5088, 1.7896])

    uhidden_biases, uhidden_weights = unnormalize_planes(m, s, hidden_biases, hidden_weights)

    plot_kwargs = {}
    quiver_kwargs = {"units": "dots", "width": 2, "headwidth": 4, "scale": 0.075, "scale_units": "dots"}

    classes = ["Apple", "Pear", "Orange"]
    labels = ["Apple boundary", "Pear boundary", "Orange boundary"]
    colors = ["greenyellow", "forestgreen", "orange"]
    linestyles = [None, "--", "-."]

    xspace = np.linspace(x_lim[0], x_lim[1], 4)
    ax1.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    ax1.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    ax1.scatter(*X[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20)

    for i, (label, color, linestyle) in enumerate(zip(labels[:2], colors[:2], linestyles[:2])):
        _ = plot_hyperplane(
            xspace,
            uhidden_biases[i],
            *uhidden_weights[i],
            5,
            c=color,
            plot_kwargs={**plot_kwargs, "label": label, "linestyle": linestyle},
            quiver_kwargs=quiver_kwargs,
            ax=ax1,
        )

    outs = uhidden_biases + X @ uhidden_weights.T
    outs = 1 / (1 + np.exp(-outs))

    ax1.set_xlabel("Weight (g)")
    ax1.set_ylabel("Diameter (cm)")
    ax1.set_aspect("equal")
    ax1.set_title("Lines for apples and pears")
    ax1.set_ylim(*y_lim)

    ax2.scatter(*outs[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    ax2.scatter(*outs[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    ax2.scatter(*outs[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20)
    outlims = get_lims(outs)
    xspace2 = np.linspace(*outlims[0], 10)
    for i, (label, color, linestyle) in enumerate(zip(labels, colors, linestyles)):
        plot_hyperplane(
            xspace2,
            output_biases[i],
            *output_weights[i],
            6,
            c=color,
            ax=ax2,
            quiver_kwargs=quiver_kwargs,
            plot_kwargs={"linestyle": linestyle, "label": label},
        )

    ax2.set_xlabel("Appleness")
    ax2.set_ylabel("Pearness")
    ax2.set_title("Activation space")
    ax2.set_xlim(*outlims[0])
    ax2.set_ylim(*outlims[1])
    ax2.set_aspect("equal")
    ax2.legend(loc="upper right")

    fig.tight_layout(rect=[0, 0, 1, 0.85])
    fig.suptitle("Visualizing appleness and pearness for each point\nwith decision boundaries in the activation space")
    plt.savefig("figures/appleness_pearness_with_out_lines.pdf")
    # plt.show()
    plt.clf()


def visualize_3lp_animated():
    hidden_biases = np.array(
        [
            -1.9472151,  # Apple
            -2.260901,  # Pear
        ]
    )

    hidden_weights = np.array(
        [
            [-4.1687274, -1.3713175],
            [4.5323997, -1.6058096],
        ]
    )

    output_biases = np.array(
        [
            -2.0450604,  # Apple
            -2.1543744,  # Pear
            2.6014535,  # Orange
        ]
    )

    output_weights = np.array(
        [
            [5.4452653, -1.87916],
            [-2.0285792, 5.59163],
            [-4.693778, -5.0045652],
        ]
    )

    m = np.array([141.8463, 6.2363])
    s = np.array([10.5088, 1.7896])

    uhidden_biases, uhidden_weights = unnormalize_planes(m, s, hidden_biases, hidden_weights)

    plot_kwargs = {}
    quiver_kwargs = {"units": "dots", "width": 2, "headwidth": 8, "scale": 0.075, "scale_units": "dots"}

    classes = ["Apple", "Pear", "Orange"]
    labels = ["Apple boundary", "Pear boundary", "Orange boundary"]
    colors = ["greenyellow", "forestgreen", "orange"]
    linestyles = [None, None, None]

    fig = plt.figure(figsize=(10, 8))
    ax_upperleft = fig.add_subplot(221)
    ax_upperright = fig.add_subplot(222)
    ax_bottom = fig.add_subplot(212)

    xspace = np.linspace(*x_lim, 4)
    ax_upperleft.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black", alpha=0.25)
    ax_upperleft.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black", alpha=0.25)
    ax_upperleft.scatter(*X[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20, alpha=0.25)

    # For use in FuncAnimation
    animated_artists = []

    hidden_lines = []
    for i, (label, color, linestyle) in enumerate(zip(labels[:2], colors[:2], linestyles[:2])):
        _, artists = plot_hyperplane(
            xspace,
            uhidden_biases[i],
            *uhidden_weights[i],
            5,
            c=color,
            plot_kwargs={**plot_kwargs, "label": label, "linestyle": linestyle},
            quiver_kwargs=quiver_kwargs,
            ax=ax_upperleft,
            return_artists=True,
        )
        hidden_lines.append(artists["line"][0])

    animated_artists.extend(hidden_lines)

    point1 = np.array([[0, 0]], dtype=float)
    animated_artists.append(
        scatterpoint1 := ax_upperleft.scatter(*point1.T, label="Unknown", marker="x", c="black", s=70, zorder=100)
    )

    h1 = uhidden_biases + X @ uhidden_weights.T
    h1 = 1 / (1 + np.exp(-h1))

    point2 = np.array([[0, 0]], dtype=float)
    animated_artists.append(
        scatterpoint2 := ax_upperright.scatter(*point2.T, label="Unknown", marker="x", c="black", s=70, zorder=100)
    )
    ax_upperright.scatter(*h1[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black", alpha=0.25)
    ax_upperright.scatter(*h1[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black", alpha=0.25)
    ax_upperright.scatter(*h1[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20, alpha=0.25)

    outlims = get_lims(h1)
    xspace2 = np.linspace(*outlims[0], 10)
    output_lines = []
    for i, (label, color, linestyle) in enumerate(zip(labels, colors, linestyles)):
        _, artists = plot_hyperplane(
            xspace2,
            output_biases[i],
            *output_weights[i],
            5,
            c=color,
            plot_kwargs={**plot_kwargs, "label": label, "linestyle": linestyle},
            quiver_kwargs=quiver_kwargs,
            ax=ax_upperright,
            return_artists=True,
        )
        output_lines.append(artists["line"][0])

    animated_artists.extend(output_lines)

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
    animated_artists.extend(chain.from_iterable(circles[1:]))
    ccenters = [[np.array(circle.get_center()) for circle in layer] for layer in circles]

    # Hidden node colors
    circles[1][0].set_facecolor(colors[0])
    circles[1][1].set_facecolor(colors[1])

    # Output node colors
    circles[2][0].set_facecolor(colors[0])
    circles[2][1].set_facecolor(colors[1])
    circles[2][2].set_facecolor(colors[2])

    # Static text in ax_bottom
    ax_bottom.annotate("Weight", ccenters[0][0] - [radius + 0.1, 0], ha="right", va="center", fontsize=fontsize)
    ax_bottom.annotate("Diameter", ccenters[0][1] - [radius + 0.1, 0], ha="right", va="center", fontsize=fontsize)
    ax_bottom.annotate("Model graph", ccenters[1][0] + [0, 0.6], ha="center", va="center", fontsize=14)
    ax_bottom.annotate("Appleness", ccenters[2][0] + [radius + 0.1, 0], ha="left", va="center", fontsize=fontsize)
    ax_bottom.annotate("Pearness", ccenters[2][1] + [radius + 0.1, 0], ha="left", va="center", fontsize=fontsize)
    ax_bottom.annotate("Orangeness", ccenters[2][2] + [radius + 0.1, 0], ha="left", va="center", fontsize=fontsize)

    # Text inside nodes
    animated_artists.extend(
        [
            x_text := ax_bottom.annotate("x", ccenters[0][0], ha="center", va="center", fontsize=fontsize),
            y_text := ax_bottom.annotate("y", ccenters[0][1], ha="center", va="center", fontsize=fontsize),
            hidden1_text := ax_bottom.annotate("h1", ccenters[1][0], ha="center", va="center", fontsize=fontsize),
            hidden2_text := ax_bottom.annotate("h2", ccenters[1][1], ha="center", va="center", fontsize=fontsize),
            out1_text := ax_bottom.annotate("o1", ccenters[2][0], ha="center", va="center", fontsize=fontsize),
            out2_text := ax_bottom.annotate("o2", ccenters[2][1], ha="center", va="center", fontsize=fontsize),
            out3_text := ax_bottom.annotate("o3", ccenters[2][2], ha="center", va="center", fontsize=fontsize),
        ]
    )

    # Current class node
    class_ccenter = np.array([1.6, 0])
    class_cradius = radius * 1.75
    ax_bottom.annotate(
        "Current classification", class_ccenter + [0, class_cradius + 0.2], ha="center", va="center", fontsize=14
    )
    animated_artists.append(class_circle := patches.Circle(class_ccenter, radius=class_cradius, **circle_kwargs))
    animated_artists.append(
        class_text := ax_bottom.annotate("Apple", class_ccenter, ha="center", va="center", fontsize=14)
    )
    ax_bottom.add_patch(class_circle)

    n = 300  # Animation steps
    pi2 = np.pi * 2
    max_linewidth = 9
    min_linewidth = 0.5
    pbar = tqdm(total=n + 1, disable=False)
    centerx = np.mean(x_lim)
    centery = np.mean(y_lim)

    def step(i):
        rad = i / n * pi2
        point1[0, 0] = centerx + 20 * math.cos(rad)
        point1[0, 1] = centery + 5 * math.sin(rad)
        scatterpoint1.set_offsets(point1)
        x_text.set_text(f"{point1[0, 0]:.1f}")
        y_text.set_text(f"{point1[0, 1]:.1f}")

        # Hidden line widths
        h1_ = uhidden_biases + point1 @ uhidden_weights.T
        h1_ = 1 / (1 + np.exp(-h1_))
        for i, line in enumerate(hidden_lines):
            line.set_linewidth(max(h1_[0, i] * max_linewidth, min_linewidth))

        hidden1_text.set_text(f"{h1_[0, 0]:.2f}")
        hidden2_text.set_text(f"{h1_[0, 1]:.2f}")
        circles[1][0].set_facecolor((*circles[1][0].get_facecolor()[:3], h1_[0, 0]))
        circles[1][1].set_facecolor((*circles[1][1].get_facecolor()[:3], h1_[0, 1]))

        scatterpoint2.set_offsets(h1_)

        # Output line widths
        out_ = output_biases + h1_ @ output_weights.T
        out_ = 1 / (1 + np.exp(-out_))
        for i, line in enumerate(output_lines):
            line.set_linewidth(max(out_[0][i] * max_linewidth, min_linewidth))

        out1_text.set_text(f"{out_[0, 0]:.2f}")
        out2_text.set_text(f"{out_[0, 1]:.2f}")
        out3_text.set_text(f"{out_[0, 2]:.2f}")
        circles[2][0].set_facecolor((*circles[2][0].get_facecolor()[:3], out_[0, 0]))
        circles[2][1].set_facecolor((*circles[2][1].get_facecolor()[:3], out_[0, 1]))
        circles[2][2].set_facecolor((*circles[2][2].get_facecolor()[:3], out_[0, 2]))

        curr_class = np.argmax(out_[0])
        class_text.set_text(classes[curr_class])
        class_circle.set_facecolor(colors[curr_class])

        pbar.update(1)
        return animated_artists

    ax_upperleft.set_xlim(*x_lim)
    ax_upperleft.set_ylim(*y_lim)
    ax_upperleft.set_aspect("equal")
    ax_upperleft.set_xlabel("Weight (g)")
    ax_upperleft.set_ylabel("Diameter (cm)")
    ax_upperleft.set_aspect("equal")
    ax_upperleft.set_title("Lines for apples and pears")

    ax_upperright.set_xlabel("Appleness")
    ax_upperright.set_ylabel("Pearness")
    ax_upperright.set_title("Activation space")
    ax_upperright.set_xlim(*outlims[0])
    ax_upperright.set_ylim(*outlims[1])
    ax_upperright.set_aspect("equal")

    ax_bottom.set_aspect("equal")
    ax_bottom.set_xlim(-3, 3)
    ax_bottom.set_ylim(-1, 1)
    ax_bottom.set_xticks([])
    ax_bottom.set_yticks([])
    ax_bottom.axis("off")

    fig.tight_layout()
    anim = FuncAnimation(fig, step, blit=True, interval=0, frames=n)
    # anim.save("figures/3lp.mp4", writer="ffmpeg", fps=60)
    plt.show()
    plt.clf()


if __name__ == "__main__":

    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    # visualize_data_set()
    # visualize_data_set_with_orange_line()
    # visualize_two_lines()
    # visualize_data_with_2lp_lines()
    # visualize_2lp_activations()
    visualize_2lp_activations_animated()
    # visualize_appleness_pearness()
    # visualize_appleness_pearness_out_lines()
    # visualize_3lp_animated()
