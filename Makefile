.PHONY: check fmt test test-cov coverage security build install

check:
	uv run ruff check src/planora/
	uv run mypy src/planora/ --strict
	uv run pytest

fmt:
	uv run ruff format src/planora/

test:
	uv run pytest

test-cov: coverage

coverage:
	uv run pytest --cov=src/planora --cov-report=term-missing

security:
	uv run bandit -r src/planora/

build:
	uv build

install:
	uv pip install -e ".[dev,tui]"
