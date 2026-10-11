PY ?= .venv/Scripts/python
SEED ?= 0
STAGE ?= 1
CFG ?= configs/base.yaml
DATA ?= ib_id
# Large caches live on D: (C: is nearly full). data/ is a junction to D:/safedti/data.
export TORCH_HOME ?= D:/safedti/torch_cache
export HF_HOME ?= D:/safedti/hf_home

.PHONY: setup verify-data test lint train eval tables

setup:
	uv venv --python 3.11 .venv
	uv pip install --python $(PY) torch --index-url https://download.pytorch.org/whl/cpu
	uv pip install --python $(PY) -e ".[dev]"

verify-data:
	$(PY) scripts/verify_interactbind.py --seed $(SEED)

test:
	$(PY) -m pytest -q

lint:
	$(PY) -m ruff format src tests scripts
	$(PY) -m ruff check src tests scripts

train:
	$(PY) -m safedti.train --stage $(STAGE) --config $(CFG) --data $(DATA) --seed $(SEED)

eval:
	$(PY) -m safedti.eval --run $(RUN)

tables:
	$(PY) scripts/make_tables.py
