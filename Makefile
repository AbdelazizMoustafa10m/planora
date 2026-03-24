.PHONY: check fmt test test-cov build install

check:
	ruff check src/planora/
	mypy src/planora/ --strict

fmt:
	ruff format src/planora/

test:
	pytest

test-cov:
	pytest --cov

build:
	uv build

install:
	uv pip install -e ".[dev,tui]"
