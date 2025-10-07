INPUT ?= .
OUT ?= output

.PHONY: install run dryrun test

install:
	pip install -r tools/monGARS_deep_scan/requirements.txt

run:
	python -m tools.monGARS_deep_scan.deep_scan --input "$(INPUT)" --out "$(OUT)"

dryrun:
	python -m tools.monGARS_deep_scan.deep_scan --dry-run --input "$(INPUT)" --out "$(OUT)"

test:
	pytest -q
