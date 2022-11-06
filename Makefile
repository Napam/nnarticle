PYTHON = python
DATA_CSV = ./data/generated/apples_oranges_pears.csv


.PHONY: all
all: $(DATA_CSV)

$(DATA_CSV): ./data/apples_oranges_pears.py
	$(PYTHON) $<

.PHONY: clean-data
clean-data:
	rm -rf ./data/generated
