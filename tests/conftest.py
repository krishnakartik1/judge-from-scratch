"""Pytest shim — load data/01_generate_candidates.py as a module.

The script's filename starts with a digit, so the normal ``import``
machinery can't reach it. Load it once per test session via
``importlib.util`` and expose the module as a top-level name that test
files can import.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "data" / "01_generate_candidates.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("stage1_gen", SCRIPT)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["stage1_gen"] = mod
    spec.loader.exec_module(mod)
    return mod


stage1_gen = _load()
