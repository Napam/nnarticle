PYTHON = python
DATA_CSV = ./data/generated/apples_oranges_pears.csv
3LP_WEIGHTS = ./models/weights/3LP.json
2LP_WEIGHTS = ./models/weights/2LP.json

FIGURES_DIR = ./visualization/figures
FIGURES_APPLES_ORANGES = \
	$(FIGURES_DIR)/dataset_apples_oranges.pdf \
	$(FIGURES_DIR)/apples_oranges_x.pdf \
	$(FIGURES_DIR)/apples_oranges_x_line.pdf
FIGURES_APPLES_ORANGES_PEARS = \
	$(FIGURES_DIR)/dataset_apples_oranges_pears.pdf \
	$(FIGURES_DIR)/apples_oranges_pears_with_apple_line.pdf \
	$(FIGURES_DIR)/apples_oranges_pears_with_hidden_lines.pdf \
	$(FIGURES_DIR)/apples_oranges_pears_2lp_lines.pdf \
	$(FIGURES_DIR)/apples_oranges_pears_2lp_activations.pdf \
	$(FIGURES_DIR)/2lp_activations.gif \
	$(FIGURES_DIR)/appleness_pearness.pdf \
	$(FIGURES_DIR)/appleness_pearness_with_out_lines.pdf \
	$(FIGURES_DIR)/3lp.pdf \
	$(FIGURES_DIR)/3lp.gif
FIGURES = $(FIGURES_APPLES_ORANGES) $(FIGURES_APPLES_ORANGES_PEARS)

.PHONY: all
all: $(3LP_WEIGHTS) $(2LP_WEIGHTS) $(FIGURES)

$(FIGURES): $(2LP_WEIGHTS) $(3LP_WEIGHTS) ./visualization/apples_oranges.py ./visualization/apples_oranges_pears.py
	$(PYTHON) ./visualization/apples_oranges.py
	$(PYTHON) ./visualization/apples_oranges_pears.py

$(2LP_WEIGHTS): $(DATA_CSV) ./models/2LP.py
	$(PYTHON) ./models/2LP.py

$(3LP_WEIGHTS): $(DATA_CSV) ./models/3LP.py
	$(PYTHON) ./models/3LP.py

$(DATA_CSV): ./data/apples_oranges_pears.py
	$(PYTHON) $<

.PHONY: clean
clean: clean-data clean-figures clean-weights

.PHONY: clean-figures
clean-figures:
	find ./visualization/figures -regex ".*\.\(mp4\|gif\|pdf\)" -delete

.PHONY: clean-data
clean-data:
	rm -rf ./data/generated

.PHONY: clean-weights
clean-weights:
	rm ./models/weights/*.json ./models/weights/*.png
