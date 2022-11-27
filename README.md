# Visualization of simple neural networks
A geometrical interpretation of neural networks

## Repository contents
This repository contains code to
1. Generate a toy dataset consisting of weight and diameter measurements of apples, oranges and pears
    1. Source file at: `./data/apples_oranges_pears.py`
    1. Generated data will be saved at: `./data/generated/apples_oranges_pears.csv`
1. Train two different neural models on the datasets, one with and without a hidden layer and store their weights as json
    1. Source files at
        1. `./models/2LP.py`
        1. `./models/3LP.py`
    1. Model weights will be saved at:
        1. `./models/weights/2LP.json`
        1. `./models/weights/3LP.json`
    1. There are also previews of the model weight boundaries adjacent to the model weight jsons

1. Visualize the dataset and model parameters
    1. Source files at:
        1. `./visualization/apples_oranges.py`
        1. `./visualization/apples_oranges_pearsjpy`
    1. Figures and animations are saved in the `./visualization/figures` directory

1. Various helper functions for the points above
    1. `./utils`
1. Some math notes related to the visualization 
    1. `./math`

## How to run
Note: dataset, model weights and figures are committed to this repository so all the generated files are already present by default

1. Ensure that you have python 3.10 or higher
1. Ensure that the packages in `requirements.txt` are installed
1. Ensure that `ffmpeg` is installed on the system and is available in your environment. That is, you should be able to run `ffmpeg -h` in your shell.
1. You can choose two methods to run the files:
    1. Execute the python files you want (see section above), this is at least how i debug and develop
    1. Run the `Makefile` (`make` needs to be available on your environment). This will create all the artifacts related to this project by executing all relevant files. As the note above states the program artifacts are already present in the repository. Running `make` should not really do anything in that case. In order to re-generate all the artifacts you need to do `make clean` first which will delete all artifacts, then `make`.
