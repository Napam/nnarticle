from pprint import pprint
import sys
import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import patches
import math
from tqdm import tqdm

from utils import get_lims, plot_hyperplane, unnormalize_plane, unnormalize_planes
plt.rcParams.update({'font.family': 'serif', 'mathtext.fontset': 'dejavuserif'})

df = pd.read_csv("datasets/apples_oranges_pears.csv")
X, y = df[["weight", "height"]].values, df["class"].map({"apple": 0, "orange": 1, "pear": 2}).values
x_lim, y_lim = get_lims(X, padding=[0.75, 1.5])


def visualize_data_set():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Comparing apples, oranges and pears")
    plt.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    plt.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    plt.scatter(*X[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20)
    plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.gca().set_aspect('equal')
    plt.gcf().set_figheight(10)
    plt.gcf().set_figwidth(10)
    plt.savefig("figures/apples_oranges_pears.pdf")
    plt.clf()


def visualize_data_set_with_orange_line():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Comparing apples, oranges and pears with a single decision boundary")
    plt.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    plt.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    plt.scatter(*X[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20)

    intercept = -0.014510540291666985
    xslope = 1.6574535
    yslope = 0.6743076

    m = [135.7327, 6.8051]
    s = [6.5687, 2.1522]

    xspace = torch.linspace(x_lim[0], x_lim[1], 4)
    intercept_, xslope_, yslope_ = unnormalize_plane(m, s, intercept, xslope, yslope)
    plot_hyperplane(xspace, intercept_, xslope_, yslope_, 10, c='k', alpha=0.75, quiver_kwargs={
                    'units': 'dots', 'width': 1.75, 'scale': 0.075, 'scale_units': 'dots'})

    plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.gca().set_aspect('equal')
    plt.gcf().set_figheight(10)
    plt.gcf().set_figwidth(10)
    plt.savefig("figures/apples_oranges_pears_with_orange_line.pdf")
    plt.clf()


def visualize_two_lines():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Decision boundaries for apples and pears")
    plt.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    plt.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    plt.scatter(*X[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20)

    intercept1 = -1.525863
    xslope1 = -4.1855865
    yslope1 = -1.2977821

    intercept2 = -1.4590645
    xslope2 = 3.8890042
    yslope2 = -1.3885064

    m = [141.8463, 6.2363]
    s = [10.5088, 1.7896]

    xspace = torch.linspace(x_lim[0], x_lim[1], 4)

    plot_kwargs = {}
    quiver_kwargs = {'units': 'dots', 'width': 1.75, 'headwidth': 4, 'scale': 0.075, 'scale_units': 'dots'}

    intercept_1, xslope_1, yslope_1 = unnormalize_plane(m, s, intercept1, xslope1, yslope1)
    plot_hyperplane(
        xspace,
        intercept_1,
        xslope_1,
        yslope_1,
        5,
        c='greenyellow',
        plot_kwargs={**plot_kwargs, 'label': 'Apple boundary'},
        quiver_kwargs=quiver_kwargs
    )

    intercept_2, xslope_2, yslope_2 = unnormalize_plane(m, s, intercept2, xslope2, yslope2)
    plot_hyperplane(
        xspace,
        intercept_2,
        xslope_2,
        yslope_2,
        5,
        c='forestgreen',
        plot_kwargs={**plot_kwargs, 'linestyle': '--', 'label': 'Pear boundary'},
        quiver_kwargs=quiver_kwargs
    )

    plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.gca().set_aspect('equal')
    plt.gcf().set_figheight(10)
    plt.gcf().set_figwidth(10)
    # plt.savefig("figures/apples_oranges_pears_two_lines.pdf")
    plt.show()
    plt.clf()


def visualize_three_lines():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Decision boundaries for apples, oranges and pears")
    plt.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    plt.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    plt.scatter(*X[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20)

    intercept1 = -1.360250473022461
    xslope1 = -3.2235553
    yslope1 = -1.1162834

    intercept2 = -1.2119554281234741
    xslope2 = 0.63510317
    yslope2 = 2.3010178

    intercept3 = -1.4134321212768555
    xslope3 = 2.7407901
    yslope3 = -1.087009

    m = [141.8463, 6.2363]
    s = [10.5088, 1.7896]

    xspace = torch.linspace(x_lim[0], x_lim[1], 4)

    plot_kwargs = {'linewidth': 3}
    quiver_kwargs = {'units': 'dots', 'width': 1.75, 'headwidth': 4, 'scale': 0.075, 'scale_units': 'dots'}

    intercept_1, xslope_1, yslope_1 = unnormalize_plane(m, s, intercept1, xslope1, yslope1)
    plot_hyperplane(
        xspace,
        intercept_1,
        xslope_1,
        yslope_1,
        8,
        c='greenyellow',
        plot_kwargs={**plot_kwargs, 'label': 'Apple boundary'},
        quiver_kwargs=quiver_kwargs
    )

    intercept_2, xslope_2, yslope_2 = unnormalize_plane(m, s, intercept2, xslope2, yslope2)
    plot_hyperplane(
        xspace,
        intercept_2,
        xslope_2,
        yslope_2,
        8,
        c='orange',
        plot_kwargs={**plot_kwargs, 'linestyle': '-.', 'label': 'Orange boundary'},
        quiver_kwargs=quiver_kwargs
    )

    intercept_3, xslope_3, yslope_3 = unnormalize_plane(m, s, intercept3, xslope3, yslope3)
    plot_hyperplane(
        xspace,
        intercept_3,
        xslope_3,
        yslope_3,
        8,
        c='forestgreen',
        plot_kwargs={**plot_kwargs, 'linestyle': '--', 'label': 'Pear boundary'},
        quiver_kwargs=quiver_kwargs
    )

    plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.gca().set_aspect('equal')
    plt.gcf().set_figheight(10)
    plt.gcf().set_figwidth(10)
    plt.savefig("figures/apples_oranges_pears_three_lines.pdf")
    # plt.show()
    plt.clf()


def visualize_likelihoods():
    plt.xlabel("Weight (g)")
    plt.ylabel("Diameter (cm)")
    plt.title("Strengths")
    # plt.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black", alpha=0.1)
    # plt.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black", alpha=0.1)
    # plt.scatter(*X[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20, alpha=0.1)

    intercepts = np.array([
        -0.08808770030736923,
        -0.09143412113189697,
        -0.09384874999523163
    ])

    slopes = np.array(
        [[-0.19972077, -0.03343868],
         [-0.021978999, 0.14851315],
         [0.20376714, -0.11762319]]
    )

    m = np.array([141.8463, 6.2363])
    s = np.array([10.5088, 1.7896])

    uintercepts, uslopes = unnormalize_planes(m, s, intercepts, slopes)

    point = np.array([[140, 6]])
    plt.scatter(*point.T, label="Unknown", marker="x", c="black", s=60)

    strengths = forward(point, uintercepts, uslopes)[0]

    xspace = torch.linspace(x_lim[0], x_lim[1], 4)

    plot_kwargs = {}
    quiver_kwargs = {'units': 'dots', 'width': 1.75, 'headwidth': 4, 'scale': 0.075, 'scale_units': 'dots'}

    linestyles = [
        None,
        '-.',
        '--'
    ]

    labels = [
        'Apple boundary',
        'Orange boundary',
        'Pear boundary'
    ]

    colors = [
        'greenyellow',
        'orange',
        'forestgreen'
    ]

    for i, (linestyle, label, color) in enumerate(zip(linestyles, labels, colors)):
        plot_hyperplane(
            xspace,
            uintercepts[i],
            *uslopes[i],
            8,
            c=color,
            plot_kwargs={**plot_kwargs, 'linestyle': linestyle, 'linewidth': max(strengths[i] * 10, 0.1), 'label': label},
            quiver_kwargs={**quiver_kwargs, 'scale': max((1 - strengths[i]) * 0.25, 0.05)}
        )

    # plt.legend(loc="upper right")
    plt.xlim(*x_lim)
    plt.ylim(*y_lim)
    plt.gca().set_aspect('equal')
    plt.gcf().set_figheight(10)
    plt.gcf().set_figwidth(10)
    plt.savefig("figures/apples_oranges_pears_strengths.pdf")
    # plt.show()
    plt.clf()


def forward(X: np.ndarray, intercepts: np.ndarray, slopes: np.ndarray):
    z = intercepts + X @ slopes.T
    z = z + abs(z.min())
    z = z ** 2
    z = z / z.sum()
    return z


def visualize_likelihoods_animated():
    intercepts = np.array([
        -0.08808770030736923,
        -0.09143412113189697,
        -0.09384874999523163
    ])

    slopes = np.array(
        [[-0.19972077, -0.03343868],
         [-0.021978999, 0.14851315],
         [0.20376714, -0.11762319]]
    )

    m = np.array([141.8463, 6.2363])
    s = np.array([10.5088, 1.7896])

    uintercepts, uslopes = unnormalize_planes(m, s, intercepts, slopes)

    plot_kwargs = {}
    quiver_kwargs = {'units': 'dots', 'width': 2, 'headwidth': 8, 'scale': 0.075, 'scale_units': 'dots'}

    classes = ['Apple', 'Orange', 'Pear']
    labels = ['Apple boundary', 'Orange boundary', 'Pear boundary']
    colors = ['greenyellow', 'orange', 'forestgreen']

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.scatter(*X[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black", alpha=0.25)
    ax1.scatter(*X[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black", alpha=0.25)
    ax1.scatter(*X[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20, alpha=0.25)

    quivers = []
    xspace = np.linspace(x_lim[0], x_lim[1], 4)
    for i, (label, color) in enumerate(zip(labels, colors)):
        _, artists = plot_hyperplane(
            xspace,
            uintercepts[i],
            *uslopes[i],
            8,
            c=color,
            plot_kwargs={**plot_kwargs, 'label': label},
            quiver_kwargs=quiver_kwargs,
            return_artists=True,
            ax=ax1
        )
        lines.append(artists['line'][0])
        quivers.append(artists['arrows'])

    point = np.array([[0, 0]], dtype=float)
    scatter = ax1.scatter(*point.T, label="Unknown", marker="x", c="black", s=60, zorder=100)
    centerx = np.mean(x_lim)
    centery = np.mean(y_lim)

    circles = []
    circletexts = []
    nnradius = 0.32
    nodesy = 0
    nodesx = -1.5
    yspacing = 0.75
    for pos, color, name in zip([(nodesx, nodesy + yspacing), (nodesx, nodesy), (nodesx, nodesy - yspacing)], colors, ['Appleness', 'Orangeness', 'Pearness']):
        circle = patches.Circle(pos, nnradius, facecolor=color, edgecolor='k')
        ax2.add_patch(circle)
        circles.append(circle)
        circletexts.append(ax2.annotate('0', xy=pos, fontsize=12, ha="center", va="center"))
        ax2.annotate(f"{name}", xy=(pos[0] + 0.5, pos[1]), fontsize=14, va="center")

    ax2.annotate("Current classification", xy=(nodesx + 3.75, nodesy + yspacing), fontsize=14, ha="center")
    currclasscircle = patches.Circle((nodesx + 3.75, nodesy), 0.6, facecolor='orange', edgecolor='k')
    currclasstext = ax2.annotate("", xy=(nodesx + 3.75, 0), ha="center", va="center", fontsize=16)
    ax2.add_patch(currclasscircle)

    ax2.annotate("Model graph", (pos[0] - 1.25, nodesy + 1.3), va="center", fontsize=14)
    xcircle = patches.Circle((nodesx - 1.25, nodesy + yspacing / 2), nnradius, facecolor=(0, 0, 0, 0), edgecolor='k')
    ycircle = patches.Circle((nodesx - 1.25, nodesy - yspacing / 2), nnradius, facecolor=(0, 0, 0, 0), edgecolor='k')
    ax2.add_patch(xcircle)
    ax2.add_patch(ycircle)
    xtext = ax2.annotate('', (nodesx - 1.25, nodesy + yspacing / 2), ha="center", va="center", fontsize=12)
    ytext = ax2.annotate('', (nodesx - 1.25, nodesy - yspacing / 2), ha="center", va="center", fontsize=12)
    ax2.annotate('Weight', (nodesx - 1.75, nodesy + yspacing / 2), ha="right", va="center", fontsize=14)
    ax2.annotate('Diameter', (nodesx - 1.75, nodesy - yspacing / 2), ha="right", va="center", fontsize=14)

    # Edges
    for left_circle in (xcircle, ycircle):
        for right_circle in circles:
            temp = np.array([left_circle.get_center(), right_circle.get_center()])
            diff = temp[1] - temp[0]
            thing = (diff / np.linalg.norm(diff)) * nnradius
            temp[0] = temp[0] + thing
            temp[1] = temp[1] - thing
            ax2.add_line(plt.Line2D(*temp.T, color='k', linewidth=1))

    n = 300
    pi2 = np.pi * 2
    pbar = tqdm(total=n)

    def step(i):
        rad = i / n * pi2
        point[0, 0] = centerx + 20 * math.cos(rad)
        point[0, 1] = centery + 5 * math.sin(rad)
        xtext.set_text(f"{point[0, 0]:.1f}")
        ytext.set_text(f"{point[0, 1]:.1f}")
        strengths = forward(point, uintercepts, uslopes)[0]
        scatter.set_offsets(point)
        for strength, line, quiver, circle, circletext in zip(strengths, lines, quivers, circles, circletexts):
            line.set_linewidth(max(strength * 20, 1))
            quiver.scale = max((1 - strength) * 0.1, 0.05)
            circletext.set_text(f"{strength:.2f}")
            circle.set_facecolor((*circle.get_facecolor()[:3], strength))

        currclass = np.argmax(strengths)
        currclasscircle.set_facecolor(colors[currclass])
        currclasstext.set_text(classes[currclass])
        pbar.update(1)

        # (not i % 20) and sys.stdout.write(f"\x1b[JAppleness: {strengths[0]}\nOrangeness: {strengths[1]}\nPearness: {strengths[2]}\n\x1b[3A")
        return (scatter, *lines, *quivers, *circletexts, *circles, currclasstext, currclasscircle, xtext, ytext)

    ax1.set_xlabel("Weight (g)")
    ax1.set_ylabel("Diameter (cm)")
    ax1.set_xlim(*x_lim)
    ax1.set_ylim(*y_lim)
    ax1.set_aspect('equal')

    ax2.axis('off')
    ax2.set_aspect('equal')
    ax2.set_xlim(-5, 5)
    ax2.set_ylim(-1.25, 1.3)

    fig.suptitle("Likelihoods of classes")
    fig.set_figheight(7)
    fig.set_figwidth(10)
    fig.tight_layout()
    anim = FuncAnimation(fig, step, blit=True, interval=0, frames=n)
    # anim.save("figures/likelihoods.gif", writer="ffmpeg", fps=24)
    plt.show()
    plt.clf()


def visualize_appleness_pearness():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))

    intercepts = np.array([
        -1.4590645, # Apple
        -1.525863,  # Pear
    ])

    slopes = np.array(
        [[3.8890042, -1.3885064],
         [-4.1855865, -1.2977821]],
    )

    m = np.array([141.8463, 6.2363])
    s = np.array([10.5088, 1.7896])

    xspace = np.linspace(x_lim[0], x_lim[1], 4)
    uintercepts, uslopes = unnormalize_planes(m, s, intercepts, slopes)

    plot_kwargs = {}
    quiver_kwargs = {'units': 'dots', 'width': 2, 'headwidth': 4, 'scale': 0.075, 'scale_units': 'dots'}

    classes = ['Pear', 'Apple']
    labels = ['Pear boundary', 'Apple boundary']
    colors = ['forestgreen', 'greenyellow']

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
            plot_kwargs={**plot_kwargs, 'label': label},
            quiver_kwargs=quiver_kwargs,
            return_artists=True,
            ax=ax1
        )

    outs = uintercepts + X @ uslopes.T
    outs[outs < 0] = 0.1 * outs[outs < 0]
    outs = outs / outs.max(0)

    ax1.set_xlabel('Weight (g)')
    ax1.set_ylabel('Diameter (cm)')
    ax1.set_aspect('equal')
    ax1.set_title('Lines for apples and pears')
    ax1.set_ylim(*y_lim)


    ax2.scatter(*outs[y == 0].T, label="Apple", marker="^", c="greenyellow", edgecolor="black")
    ax2.scatter(*outs[y == 1].T, label="Orange", marker="o", c="orange", edgecolor="black")
    ax2.scatter(*outs[y == 2].T, label="Pear", marker="s", c="forestgreen", edgecolor="black", s=20)
    ax2.set_xlabel("Pearness")
    ax2.set_ylabel("Appleness")
    ax2.set_title("Appleness and pearness")
    ax2.set_aspect('equal')

    fig.tight_layout(rect=[0,0,1,0.90])
    fig.suptitle("Visualizing appleness and pearness for each point")
    plt.savefig('figures/appleness_pearness.pdf')
    # plt.show()
    plt.clf()



if __name__ == '__main__':
    # visualize_data_set_with_orange_line()
    # visualize_data_set()
    # visualize_two_lines()
    # visualize_three_lines()
    # visualize_likelihoods()
    # visualize_likelihoods_animated()
    visualize_appleness_pearness()
