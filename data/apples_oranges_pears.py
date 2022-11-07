import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from matplotlib import pyplot as plt
import pathlib
import os
import sys
import logging

rng = np.random.default_rng(2)

logger = logging.getLogger("data.apples_oranges_pears")


def make_blobs(sizes: ArrayLike, means: ArrayLike, stds: ArrayLike):
    X = np.concatenate([rng.standard_normal((size, 2)) * std + mean for size, mean, std in zip(sizes, means, stds)])
    y = np.arange(len(sizes)).repeat(sizes)

    shufflerState = np.random.default_rng(1)
    shufflerState.shuffle(X)
    shufflerState = np.random.default_rng(1)
    shufflerState.shuffle(y)

    logger.debug(f"X=\n{X}")
    logger.debug(f"y=\n{y}")
    return X, y


if __name__ == "__main__":
    outfile = pathlib.Path(os.path.dirname(os.path.abspath(__file__))) / "generated" / "apples_oranges_pears.csv"
    logging.basicConfig(level=logging.INFO)

    if outfile.exists():
        logger.info(f"{outfile} already exists, will not generate anything")
        sys.exit()

    outfile.parent.mkdir(exist_ok=True)

    avg_orange_diameter = 8  # cm
    avg_orange_weight = 140  # gram

    avg_apple_diameter = 6  # cm
    avg_apple_weight = 130  # gram

    avg_pear_diameter = 5  # cm
    avg_pear_weight = 155  # gram

    centers = [
        [avg_apple_weight, avg_apple_diameter],
        [avg_orange_weight, avg_orange_diameter],
        [avg_pear_weight, avg_pear_diameter],
    ]

    X, y = make_blobs([30] * len(centers), centers, [[4.5, 1.75]] * len(centers))

    df = pd.DataFrame(
        {"weight": X[:, 0], "height": X[:, 1], "class": np.vectorize({0: "apple", 1: "orange", 2: "pear"}.get)(y)}
    )

    df.to_csv(outfile, index=False)
    logger.info(f"Generated {outfile}")