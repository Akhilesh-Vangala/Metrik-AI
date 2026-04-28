.PHONY: all setup data cython test run run-dev benchmark parallel-benchmark spark-benchmark compare profile spark quality eda clean

PYTHON = python3
CONFIG = config/config.yaml

all: run eda benchmark compare parallel-benchmark profile test

setup:
	pip install -r requirements.txt
	$(MAKE) cython

data:
	$(PYTHON) scripts/download_data.py

cython:
	$(PYTHON) setup.py build_ext --inplace

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

run:
	$(PYTHON) -m src.cli --config $(CONFIG) run

run-dev:
	$(PYTHON) -m src.cli --config $(CONFIG) run --n-chunks 6

benchmark:
	$(PYTHON) -m src.cli --config $(CONFIG) benchmark

parallel-benchmark:
	$(PYTHON) -m src.cli --config $(CONFIG) parallel-benchmark

compare:
	$(PYTHON) -m src.cli --config $(CONFIG) compare

profile:
	$(PYTHON) -m src.cli --config $(CONFIG) profile

spark:
	$(PYTHON) -m src.cli --config $(CONFIG) spark

spark-benchmark:
	$(PYTHON) -m src.cli --config $(CONFIG) spark-benchmark

quality:
	$(PYTHON) -m src.cli --config $(CONFIG) quality

eda:
	$(PYTHON) -m src.cli --config $(CONFIG) eda

clean:
	rm -rf build/ dist/ *.egg-info
	rm -f src/*.so src/*.c
	rm -rf results/profiling/
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	find . -name "*.pyc" -delete
