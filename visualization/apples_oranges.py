import torch
import pandas as pd
from matplotlib import pyplot as plt
import sys
from pathlib import Path
import logging

project_dir = Path(__file__).resolve().parent.parent
figures_dir = project_dir / "visualization" / "figures"
sys.path.insert(0, str(project_dir))

from utils import get_lims, plot_hyperplane, unnormalize_planes, json_to_weights, setup_pyplot_params

logger = logging.getLogger("visualize.apples_oranges")

setup_pyplot_params()

try:
    df = pd.read_csv(project_dir / "data" / "generated" / "apples_oranges_pears.csv")
except Exception as e:
    logger.error(f"Something wrong when attempting to import data: {e}")
    sys.exit()

X, y = df[["weight", "height"]].values, df["class"].map({"apple": 0, "orange": 1, "pear": 2}).values
x_lim, y_lim = get_lims(df[~df["class"].str.fullmatch("pear")][["weight", "height"]].values, padding=1.25)


def savefig(file: str | Path):
    file = figures_dir / file
    plt.savefig(file)
    logger.info(f"Created figure {file}")


def visualize_data_set(save: bool = True, clf: bool = True):
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Comparing apples and oranges")
    plt.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    plt.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.gca().set_aspect("equal")
    plt.gcf().set_figheight(10)
    plt.gcf().set_figwidth(10)

    if save:
        savefig("dataset_apples_oranges.pdf")

    # plt.show()

    if clf:
        plt.clf()


def visualize_data_set_with_unknown_point(save: bool = True, clf: bool = True):
    visualize_data_set(False, False)

    plt.title("Comparing apples and oranges with an unknown")
    plt.scatter([130], [5.5], label="Unknown", marker="x", c="black", s=80)

    if save:
        savefig("apples_oranges_x.pdf")

    # plt.show()

    if clf:
        plt.clf()


def visualize_data_set_with_unknown_point_and_line():
    visualize_data_set_with_unknown_point(False, False)

    m = X.mean(0)
    s = X.std(0)

    model_weights = json_to_weights(project_dir / "models" / "weights" / "3LP.json")
    intercepts = model_weights["hidden.bias"]
    weights = model_weights["hidden.weight"]
    xspace = torch.linspace(x_lim[0], x_lim[1], 4)
    uintercepts, uweights = unnormalize_planes(m, s, intercepts, weights)

    plot_hyperplane(
        xspace,
        uintercepts[0],
        uweights[0, 0],
        uweights[0, 1],
        6,
        c="k",
        quiver_kwargs={"units": "dots", "width": 2.25, "scale": 0.065, "scale_units": "dots"},
    )

    savefig("apples_oranges_x_line.pdf")

    # plt.show()
    plt.clf()


if __name__ == "__main__":

    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    visualize_data_set()
    visualize_data_set_with_unknown_point()
    visualize_data_set_with_unknown_point_and_line()
