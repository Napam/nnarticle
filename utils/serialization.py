import json
import torch
from torch import nn
from pathlib import Path


def model_to_json(model: nn.Module, file: str | Path = None, indent: int = 2, **kwargs) -> str | None:
    serializable_state_dict = {k: v.numpy().tolist() for k, v in model.state_dict().items()}
    jsonstr = json.dumps(serializable_state_dict, indent=indent, **kwargs)

    if file is None:
        return jsonstr

    file = Path(file)
    file.parent.mkdir(parents=True, exist_ok=True)
    with open(file, "w") as f:
        f.write(jsonstr)

    return jsonstr


if __name__ == "__main__":

    class Test(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(2, 4)

    model = Test()
    print(model_to_json(model, indent=2))
