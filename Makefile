.PHONY: setup data cython test run benchmark profile spark clean

PYTHON = python
CONFIG = config/config.yaml

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
	$(PYTHON) -m src.cli --config $(CONFIG) run --n-chunks 3

benchmark:
	$(PYTHON) -m src.cli --config $(CONFIG) benchmark

profile:
	$(PYTHON) -m src.cli --config $(CONFIG) profile

spark:
	$(PYTHON) -m src.cli --config $(CONFIG) spark

clean:
	rm -rf build/ dist/ *.egg-info
	rm -f src/*.so src/*.c
	rm -rf results/profiling/
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	find . -name "*.pyc" -delete
