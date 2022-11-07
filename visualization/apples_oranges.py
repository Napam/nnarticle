import torch
import pandas as pd
from matplotlib import pyplot as plt
import sys
from pathlib import Path
import logging

project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

from utils import get_lims, plot_hyperplane, unnormalize_planes

logger = logging.getLogger("visualize.apples_and_oranges")

plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})

try:
    df = pd.read_csv(project_dir / 'data' / 'generated' / 'apples_oranges_pears.csv')
except Exception as e:
    logger.error(f"Something wrong when attempting to import data: {e}")
    sys.exit()

df = df[~df['class'].str.fullmatch("pear")]
X, y = df[["weight", "height"]].values, df["class"].map({"apple": 0, "orange": 1}).values
print(y.shape)
x_lim, y_lim = get_lims(X, padding=1.25)


def visualize_data_set():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Comparing apples and oranges")
    plt.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    plt.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.gca().set_aspect('equal')
    plt.gcf().set_figheight(10)
    plt.gcf().set_figwidth(10)

    file = project_dir / 'visualization' / 'figures' / 'dataset_apples_oranges_pears.pdf'
    plt.savefig(file)
    logger.info(f'Created figure {file}')

    plt.clf()


def visualize_data_set_with_unknown_point():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Comparing apples and oranges with an unknown")
    plt.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    plt.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    plt.scatter([130], [5.5], label="Unknown", marker="x", c="black", s=80)
    plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.gca().set_aspect('equal')
    plt.gcf().set_figheight(10)
    plt.gcf().set_figwidth(10)

    file = project_dir / 'visualization' / 'figures' / 'apples_oranges_x.pdf'
    plt.savefig(file)
    logger.info(f'Created figure {file}')

    plt.clf()


def visualize_data_set_with_unknown_point_and_line():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Comparing apples and oranges with an unknown and a decision boundary")
    plt.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    plt.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    plt.scatter([130], [5.5], label="Unknown", marker="x", c="black", s=80)

    intercept = -0.014510540291666985
    xslope = 1.6574535
    yslope = 0.6743076

    m = [135.7327, 6.8051]
    s = [6.5687, 2.1522]

    xspace = torch.linspace(x_lim[0], x_lim[1], 4)
    intercept_, xslope_, yslope_ = unnormalize_plane(m, s, intercept, xslope, yslope)
    plot_hyperplane(xspace, intercept_, xslope_, yslope_, 5, c='k', alpha=0.75, quiver_kwargs={
                    'units': 'dots', 'width': 1.75, 'scale': 0.075, 'scale_units': 'dots'})
    plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.gca().set_aspect('equal')
    plt.gcf().set_figheight(10)
    plt.gcf().set_figwidth(10)
    plt.savefig("figures/applesnoranges_unknown_point_with_line.pdf")
    plt.clf()


if __name__ == "__main__":

    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    visualize_data_set()
    visualize_data_set_with_unknown_point()
    visualize_data_set_with_unknown_point_and_line()
