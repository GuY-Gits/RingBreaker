# RingBreaker — one-command workflows. Requires Python >= 3.10 and Node >= 18.
PY ?= .venv/bin/python
PORT ?= 8001

.PHONY: setup data api dev dashboard build test test-py test-web smoke demo

setup:            ## create venv, install backend + dashboard deps
	test -d .venv || python3 -m venv .venv
	$(PY) -m pip install -q -e ".[dev]"
	cd dashboard && npm ci

data:             ## regenerate simulator data, retrain models, write evaluation
	$(PY) -m ringbreaker.pipeline

build:            ## build the dashboard (served by the API at /app)
	cd dashboard && npm run build

api:              ## run the API (+ built dashboard) on :$(PORT)
	$(PY) -m uvicorn ringbreaker.api.main:app --port $(PORT)

dashboard:        ## dashboard dev server with hot reload on :5173 (proxies /api)
	cd dashboard && BACKEND_PORT=$(PORT) npm run dev

demo: build api   ## build the dashboard, then serve everything at http://127.0.0.1:$(PORT)/app

test: test-py test-web

test-py:
	$(PY) -m pytest -q

test-web:
	cd dashboard && npm run typecheck && npm test

smoke:            ## end-to-end check against a running API
	$(PY) scripts/smoke_test.py --url http://127.0.0.1:$(PORT)
