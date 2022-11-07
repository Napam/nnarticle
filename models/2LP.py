import torch
import numpy as np
import pandas as pd
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
from torch.nn.functional import one_hot
import sys
from pathlib import Path
import logging

project_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_dir))

from utils import get_lims, normalize_data, plot_hyperplane, unnormalize_planes, model_to_json

logger = logging.getLogger("models.2LP")


def mse(y_: torch.Tensor, y: torch.Tensor):
    assert y_.shape == y.shape
    return ((y_ - y) ** 2).mean()


def accuracy(y_: torch.Tensor, y: torch.Tensor):
    y_, y = y_.detach(), y.detach()
    y_ = y_.argmax(1)
    y = y.argmax(1)
    return (y_ == y).sum() / len(y)


class TwoLayerPerceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.planes = nn.Linear(in_features=2, out_features=3)

    def forward(self, X: torch.Tensor):
        return torch.sigmoid(self.planes(X))

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        optimizer = optim.Adam(self.parameters(), lr=3e-3, weight_decay=0.1)
        criterion = mse

        losses = []
        for i in range(1000):
            y_ = self.forward(X)
            loss = criterion(y_, y)
            logger.debug(f"Loss: {loss.item():<25} Accuracy: {accuracy(y_, y).item()}")
            if losses and torch.allclose(losses[-1], loss, atol=5e-7):
                logger.info(f"Achieved satisfactory loss convergence at {loss}")
                break
            losses.append(loss.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            logger.info(f"Hit maximum iteration at loss {loss}")

    def plot(self, X: torch.Tensor, y: torch.Tensor, X_mean: torch.Tensor, X_std: torch.Tensor):
        X = X.numpy()
        y = y.numpy()
        X_mean = X_mean.numpy()
        X_std = X_std.numpy()

        X = X * X_std + X_mean
        biases, weights = self.planes.bias.detach().numpy(), self.planes.weight.detach().numpy()

        colors = ["greenyellow", "orange", "forestgreen"]
        plt.scatter(*X.T, c=np.vectorize({i: c for i, c in enumerate(colors)}.get)(y.argmax(1)))

        xspace = np.array([X[:, 0].min() * 0.75, X[:, 0].max() * 1.25])
        ubiases, uweights = unnormalize_planes(X_mean, X_std, biases, weights)
        for color, ubias, uweight in zip(colors, ubiases, uweights):
            plot_hyperplane(
                xspace,
                ubias,
                *uweight,
                16,
                c=color,
                quiver_kwargs={"scale": 0.05, "units": "dots", "width": 2},
            )

        xlim, ylim = get_lims(X, padding=0.5)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.gca().set_aspect("equal")


if __name__ == "__main__":

    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.setLevel(logging.INFO)

    try:
        df = pd.read_csv(project_dir / 'data' / 'generated' / 'apples_oranges_pears.csv')
    except Exception as e:
        logger.error(f"Something wrong when attempting to import data: {e}")
        sys.exit()

    model = TwoLayerPerceptron()

    X_raw = torch.tensor(df[["weight", "height"]].values, dtype=torch.float32)
    y_raw = torch.tensor(df["class"].map({"apple": 0, "orange": 1, "pear": 2}).values, dtype=torch.long)

    X, X_mean, X_std = normalize_data(X_raw)
    y = one_hot(y_raw)

    model.fit(X, y)

    stem = Path(__file__).stem
    weight_file = project_dir / 'models' / 'weights' / f'{stem}.json'
    image_file = project_dir / 'models' / 'weights' / f'{stem}.png'

    if weight_file.exists():
        logger.info(f'Weight file for {__file__} already exists, will not save')
        sys.exit()

    model_to_json(model, weight_file)
    logger.info(f'Saved weight file for {__file__} at {weight_file}')
    model.plot(X, y, X_mean, X_std)
    plt.savefig(image_file)
    logger.info(f'Saved weight image file for {__file__} at {image_file}')

