DATA_DIR := data
VENV := venv
PIP := $(VENV)/bin/pip

.PHONY: setup install run clean

setup:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	make install

install:
	$(PIP) install -r requirements.txt

run:
	. venv/bin/activate && python src/main.py

clean:
	rm -rf venv

download-dataset:
	mkdir -p $(DATA_DIR)
	cd $(DATA_DIR) && wget https://mcfp.felk.cvut.cz/publicDatasets/CTU-13-Dataset/CTU-13-Dataset.tar.bz2
	cd $(DATA_DIR) && tar -xjf CTU-13-Dataset.tar.bz2

download-scenario:
	mkdir -p $(DATA_DIR)
	cd $(DATA_DIR) && wget https://mcfp.felk.cvut.cz/publicDatasets/CTU-Malware-Capture-Botnet-$(SCENARIO)
	cd $(DATA_DIR) && tar -xjf CTU-Malware-Capture-Botnet-$(SCENARIO).tar.bz2