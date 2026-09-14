# Local workbench commands. Research computations are intentionally not part
# of the default verification path.

PYTHON ?= python3.12
VENV   ?= .venv
PY     := $(VENV)/bin/python

.PHONY: setup test test-d2 smoke verify

setup:
	$(PYTHON) -m venv $(VENV)
	$(PY) -m pip install -r requirements_lock.txt

test:
	PYTHONPATH=src $(PY) -m pytest src/simulator/tests -q

test-d2:
	@test -n "$(KONARK_REPO_ROOT)" || \
	  (echo "Set KONARK_REPO_ROOT to a konqr/lobSimulations checkout." && exit 2)
	PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src KONARK_REPO_ROOT="$(KONARK_REPO_ROOT)" \
	  $(PY) -m pytest \
	  src/simulator/tests/test_d2_konark_backend.py \
	  src/simulator/tests/test_d2_konark_latent_book.py \
	  src/simulator/tests/test_d2_inference_chain.py \
	  src/simulator/tests/test_d2_regime_detectability.py -q

smoke:
	PYTHONPATH=src $(PY) -m simulator \
	  --config configs/first_milestone.json \
	  --seed 1 --horizon 5 --observe --summarise \
	  --runs-root .workbench/smoke

verify: test smoke
