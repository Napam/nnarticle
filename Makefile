PYTHON = python
DATA_CSV = ./data/generated/apples_oranges_pears.csv
3LP_WEIGHTS = ./models/weights/3LP.json
2LP_WEIGHTS = ./models/weights/2LP.json


.PHONY: all
all: $(3LP_WEIGHTS) $(2LP_WEIGHTS)

$(2LP_WEIGHTS): $(DATA_CSV)
	$(PYTHON) ./models/2LP.py

$(3LP_WEIGHTS): $(DATA_CSV)
	$(PYTHON) ./models/3LP.py

$(DATA_CSV): ./data/apples_oranges_pears.py
	$(PYTHON) $<

.PHONY: clean-data
clean-data:
	rm -rf ./data/generated

.PHONY: clean-weights
clean-weights:
	rm ./models/weights/*.json ./models/weights/*.png
