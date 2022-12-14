"""
Helper functions for serialization / deserialization of nn.Module instances
"""

import json
import numpy as np
import torch
from torch import nn
from pathlib import Path
from io import TextIOBase


def model_to_json(model: nn.Module, file: str | Path = None, indent: int = 2, **kwargs) -> str:
    """
    Serialize given PyTorch nn.Module model to JSON.

    Parameters
    ----------
    model : nn.Module
        PyTorch model

    file : str or Path, optional
        If given a path, store the model

    indent: int, default = 2
        Indentation of JSON file

    **kwargs : dict, optional
        Extra arguemnts to json.dumps function

    Returns
    -------
    str
        Model as serialized JSON
    """

    serializable_state_dict = {k: v.numpy().tolist() for k, v in model.state_dict().items()}
    jsonstr = json.dumps(serializable_state_dict, indent=indent, **kwargs)

    if file is None:
        return jsonstr

    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w") as f:
        f.write(jsonstr)

    return jsonstr


def json_to_weights(file: str | Path | TextIOBase) -> dict[str, np.ndarray]:
    """
    Reads serialized model and produces a dictionary of model weights as numpy arrays

    Parameters
    ----------
    file : str or Path or TextIOBase
        path to json file

    Returns
    -------
    dict where key are layer names and values are layer values
    """

    close_file_after = False
    if not isinstance(file, TextIOBase):
        close_file_after = True
        file = open(Path(file), "r")

    weights = json.load(file)

    if close_file_after:
        file.close()

    return {k: np.array(v) for k, v in weights.items()}


if __name__ == "__main__":
    import io
    from pprint import pformat

    class Test(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(2, 3)
            self.layer2 = nn.Linear(3, 3)

    model = Test()
    serialized = model_to_json(model, indent=2)
    print(f"serialized={serialized}")

    testfile = io.StringIO()
    testfile.write(serialized)
    testfile.seek(0)

    print(f"deserialized={pformat(json_to_weights(testfile))}")
