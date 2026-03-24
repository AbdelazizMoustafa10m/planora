"""Allow `python -m planora`."""

import planora.cli  # noqa: F401 — registers all CLI sub-module commands
from planora.cli.app import app

app()
