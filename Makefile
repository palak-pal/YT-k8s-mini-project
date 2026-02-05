.PHONY: install run test

VENV ?= .venv
PYTHON ?= python3
PIP := $(VENV)/bin/pip
STREAMLIT := $(VENV)/bin/streamlit
PYTEST := $(VENV)/bin/pytest

$(VENV)/.installed: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt
	touch $@

install: $(VENV)/.installed

run: install
	$(STREAMLIT) run app/Home.py

test: install
	$(PYTEST) -q
