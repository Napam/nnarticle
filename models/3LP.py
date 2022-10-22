import torch
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
from torch.nn.functional import one_hot, cross_entropy, softmax
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import get_lims, normalize_data, plot_hyperplane, unnormalize_plane, unnormalize_planes

def accuracy(y_: torch.Tensor, y: torch.Tensor):
    y_, y = y_.detach(), y.detach()
    y_ = y_.argmax(1)
    y = y.argmax(1)
    return (y_ == y).sum() / len(y)


def mse(y_: torch.Tensor, y: torch.Tensor):
    assert y_.shape == y.shape
    return ((y_ - y) ** 2).mean()


class ThreeLayerPerceptron(nn.Module):

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(in_features=2, out_features=2)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)
        self.output = nn.Linear(in_features=2, out_features=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.Tensor, return_z1: bool = False):
        z1 = self.hidden(X)
        h1 = self.leaky_relu(z1)
        z2 = self.output(h1)
        h2 = self.sigmoid(z2)
        out = h2 / h2.sum(1).reshape(-1, 1)
        if return_z1:
            return h2, z1
        return h2

    def conditioner_criterion(self, y_, y, z1):
        h = torch.sigmoid(z1)
        y_h_ = torch.column_stack([h[:, 0], (h < 0.5).all(1) * (1 - h.sum(1) / 2), h[:, 1]])

        return cross_entropy(y_, y) + 3 * mse(y_h_, y)

    def fit(self, X: torch.tensor, y: torch.tensor):
        y = y.to(torch.float32)
        optimizer = optim.Adam(self.parameters(), lr=4e-3, weight_decay=0.001)
        criterion = self.conditioner_criterion

        losses = []
        for i in range(10000):
            y_, z1 = self.forward(X, True)
            loss = criterion(y_, y, z1)
            print(f"Loss: {loss.item():<25} Accuracy: {accuracy(y_, y).item()}")
            if losses and torch.allclose(losses[-1], loss, atol=5e-7):
                print("Achieved satisfactory loss convergence")
                break
            losses.append(loss.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            print("Hit maximum iteration")

    def plot(self, X: torch.Tensor, y: torch.Tensor, X_mean: torch.Tensor, X_std: torch.Tensor):
        X = X.numpy()
        y = y.numpy()
        X_mean = X_mean.numpy()
        X_std = X_std.numpy()

        X = X * X_std + X_mean
        xspace1 = np.array([X[:, 0].min() * 0.75, X[:, 0].max() * 1.25])

        hidden_biases, hidden_weights = self.hidden.bias.detach().numpy(), self.hidden.weight.detach().numpy()
        uintercepts, uslopes = unnormalize_planes(X_mean, X_std, hidden_biases, hidden_weights)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        ax1.scatter(*X.T, c=y)
        colors = ['red', 'green', 'blue']
        for i, color in enumerate([colors[0], colors[2]]):
            plot_hyperplane(xspace1, uintercepts[i], uslopes[i][0], uslopes[i][1], 10, c=color, quiver_kwargs={'scale': 0.05, 'units': 'dots', 'width': 2}, ax=ax1)


        X_hidden = self.leaky_relu(torch.tensor(uintercepts + X @ uslopes.T)).detach().numpy()
        ax2.scatter(*X_hidden.T, c=y)
        output_biases, output_weights = self.output.bias.detach().numpy(), self.output.weight.detach().numpy()
        xspace2 = np.array([X_hidden[:, 0].min() * 0.75, X_hidden[:, 0].max() * 1.25])

        for i, color in enumerate(colors[:3]):
            plot_hyperplane(xspace2, output_biases[i], *output_weights[i], 10, c=color, quiver_kwargs={'scale': 0.05, 'units': 'dots', 'width': 2}, ax=ax2)


        xlim1, ylim1 = get_lims(X, padding=0.5)
        ax1.set_xlim(*xlim1)
        ax1.set_ylim(*ylim1)
        ax1.set_aspect('equal')

        xlim2, ylim2 = get_lims(X_hidden, padding=0.1)
        ax2.set_xlim(*xlim2)
        ax2.set_ylim(*ylim2)
        ax2.set_aspect('equal')

        plt.show()

        print('hidden_biases:', hidden_biases)
        print('hidden_weights:', hidden_weights)
        print('output_biases:', output_biases)
        print('output_weights:', output_weights)


if __name__ == '__main__':
    model = ThreeLayerPerceptron()

    df = pd.read_csv("../datasets/apples_oranges_pears.csv")
    X_raw = torch.tensor(df[["weight", "height"]].values, dtype=torch.float32)
    y_raw = torch.tensor(df["class"].map({"apple": 0, "orange": 1, "pear": 2}).values, dtype=torch.long)

    X, X_mean, X_std = normalize_data(X_raw)
    y = one_hot(y_raw)

    model.fit(X, y)
    model.plot(X, y, X_mean, X_std)

