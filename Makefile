SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c

ENV_NAME ?= env-continuouspersuasion
VENV_DIR ?= $(ENV_NAME)

PYTHON_VENV := $(VENV_DIR)/bin/python
PYTHON_VENV_ABS := $(abspath $(PYTHON_VENV))
PIP_VENV := $(VENV_DIR)/bin/pip
CONDA_RUN := conda run -n $(ENV_NAME)
NPM_INSTALL_CMD := if [ -f package-lock.json ]; then npm ci; else npm install; fi

# Fast, cached pylint configuration
PYLINTHOME ?= $(abspath .pylint_cache)
# Default fast flags: parallelize across CPUs and enable cache persistence
PYLINT_FLAGS ?= -j 0 --persistent=yes

.PHONY: init init-api init-r pytest jsbuild jslint pylint pylint-errors pylint-all-errors release-% jstest serve get-data Rfmt Rlint paper-assets

init: init-r
	python3.11 -m venv env-continuouspersuasion
	$(PIP_VENV) install -r requirements.txt
	@if [ -f requirements-mechanism-rl.txt ]; then \
		$(PIP_VENV) install -r requirements-mechanism-rl.txt; \
	fi
	$(PYTHON_VENV) -m ipykernel install --user --name "env-continuouspersuasion"
	$(PIP_VENV) install --editable .

init-api:
	python3.11 -m venv env-continuouspersuasion
	$(PIP_VENV) install -r requirements-api.txt
	$(PIP_VENV) install --editable . --no-deps

init-r:
	Rscript -e 'if (!requireNamespace("renv", quietly = TRUE)) install.packages("renv")'
	@if [ ! -f renv.lock ]; then Rscript -e 'renv::init(bare = TRUE)'; fi
	Rscript -e 'renv::install("lme4")'
	Rscript -e 'renv::install("lmerTest")'
	Rscript -e 'renv::install("emmeans")'
	Rscript -e 'renv::install("pbkrtest")'
	Rscript -e 'renv::install("styler")'
	Rscript -e 'renv::install("lintr")'
	Rscript -e 'renv::snapshot()'

release-%:    ## e.g. `make release-continouspersuasion` or `make release-continouspersuasion-public`
	git pull
	$(MAKE) jsbuild
	- mv -f database.db database_old.db
	- rm -f logs/main.log
	sudo systemctl restart $(@:release-%=%)

serve:
	$(MAKE) jsbuild
	$(PYTHON_VENV) -m fastapi dev src/main.py

init-conda:
	conda env create --file environment.yml
	$(CONDA_RUN) pip install --editable ".[mechanism-rl]"

jsbuild:
	$(NPM_INSTALL_CMD)
	# Build the frontend workspace
	npm run -w frontend build

jslint:
	# Ensure root devDependencies (eslint, prettier, plugins) are installed
	@if [ ! -x ./node_modules/.bin/eslint ] || [ ! -x ./node_modules/.bin/prettier ]; then \
		$(NPM_INSTALL_CMD); \
	fi
	./node_modules/.bin/prettier --cache --write frontend/src analysis/static
	./node_modules/.bin/eslint --cache --fix frontend/src analysis/static
	@if [ ! -x ./node_modules/.bin/jscpd ]; then \
		$(NPM_INSTALL_CMD); \
	fi
	./node_modules/.bin/jscpd --gitignore --pattern '**/*.{js,mjs,cjs}'

mdfmt:
	# Ensure Prettier is available
	@if [ ! -x ./node_modules/.bin/prettier ]; then \
		$(NPM_INSTALL_CMD); \
	fi
	./node_modules/.bin/prettier --write '**/*.md'
	@if [ ! -x ./node_modules/.bin/jscpd ]; then \
		$(NPM_INSTALL_CMD); \
	fi
	./node_modules/.bin/jscpd --gitignore --pattern '**/*.md'

# Code formatting: isort (imports) + black (code style)
pyfmt:
	$(PYTHON_VENV) -m isort --profile black src analysis
	$(PYTHON_VENV) -m black src analysis

pylint:
	@pylint_status=0; \
	$(PYTHON_VENV) -m pylint $(PYLINT_FLAGS) src analysis || pylint_status=$$?; \
	$(PYTHON_VENV) -m vulture src analysis; \
	if [ ! -x ./node_modules/.bin/jscpd ]; then \
		$(NPM_INSTALL_CMD); \
	fi; \
	./node_modules/.bin/jscpd --gitignore --pattern '**/*.py' --reporters consoleFull; \
	exit $$pylint_status

# Quick pass: errors only (useful locally)
pylint-errors:
	@pylint_status=0; \
	$(PYTHON_VENV) -m pylint $(PYLINT_FLAGS) --errors-only src analysis || pylint_status=$$?; \
	exit $$pylint_status

# PYTHONPATH=src so that the working directory for the commands is the top level directory above the package.
# This might be hacky
# For local resource-light tests
pytest:
	$(PYTHON_VENV) -m pytest

jstest:
	npm run -w frontend test:unit

get-data:
	scp jared_jaredmoore_org@35.208.218.204:/opt/continuouspersuasion/logs/main.log main.log
	scp jared_jaredmoore_org@35.208.218.204:/opt/continuouspersuasion/database.db database.db	

Rfmt:
	@echo "Styling R scripts under analysis/ with styler..."
	Rscript -e "styler::style_dir('analysis', filetype = 'R')"

Rlint:
	@echo "Linting R scripts under analysis/ with lintr..."
	Rscript -e "res <- lintr::lint_dir('analysis'); print(res); if (length(res) > 0L) quit(status = 1) else quit(status = 0)"

paper-assets:
	$(PYTHON_VENV) -m analysis.latex.export_paper_assets
