"""Pytest shim — load digit-prefixed data/*.py scripts as modules.

The stage scripts' filenames start with a digit, so the normal
``import`` machinery can't reach them. Load each once per test session
via ``importlib.util`` and expose the module as a top-level name that
test files can import.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load(module_name: str, script_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


stage1_gen = _load("stage1_gen", REPO_ROOT / "data" / "01_generate_candidates.py")
stage1_enrich = _load("stage1_enrich", REPO_ROOT / "data" / "01b_enrich_candidates.py")
stage2_pairs = _load("stage2_pairs", REPO_ROOT / "data" / "02_construct_pairs.py")
stage3a_holdout = _load("stage3a_holdout", REPO_ROOT / "data" / "03a_holdout_eval.py")
stage4_label = _load("stage4_label", REPO_ROOT / "data" / "04_label_pairs.py")
