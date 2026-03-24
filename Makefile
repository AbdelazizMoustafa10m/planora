.PHONY: check fmt lint typecheck security test test-cov coverage build install

check: fmt lint typecheck security test

fmt:
	uv run ruff format src/planora/

lint:
	uv run ruff check src/planora/

typecheck:
	uv run mypy src/planora/ --strict

test:
	uv run pytest

security:
	uv run bandit -r src/planora/

test-cov: coverage

coverage:
	uv run pytest --cov=src/planora --cov-report=term-missing

build:
	uv build

install:
	uv pip install -e ".[dev,tui]"
