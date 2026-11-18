"""Every example script must at least import (catches syntax errors and broken imports without running them)."""

import importlib
from pathlib import Path

import pytest

EXAMPLES = sorted(p.stem for p in (Path(__file__).resolve().parents[2] / "examples").glob("*.py") if p.stem != "__init__")


@pytest.mark.parametrize("name", EXAMPLES)
def test_example_imports(name):
    module = importlib.import_module(f"examples.{name}")
    assert callable(module.main)
