.PHONY: install run dryrun test

PY ?= python3
OUT ?= output
INPUT ?= .

# Adjust MODULE if your package lives elsewhere
MODULE ?= tools.monGARS_deep_scan
REQFILE ?= tools/monGARS_deep_scan/requirements.txt

install:
$(PY) -m pip install -r $(REQFILE)

run:
PYTHONPATH=. $(PY) -m $(MODULE).deep_scan --input $(INPUT) --out $(OUT)

dryrun:
PYTHONPATH=. $(PY) -m $(MODULE).deep_scan --input $(INPUT) --out $(OUT) --dry-run

test:
pytest -q
