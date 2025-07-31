VENV = $(HOME)/aimemory_cpu
PY   = $(VENV)/bin/python

venv:
	python3 -m venv $(VENV)

install: venv
	$(VENV)/bin/pip install -q -r requirements.txt

revector:
	$(PY) maintenance/luna_revector.sh
