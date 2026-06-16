"""Pytest bootstrap: make the ``src/`` package layout importable.

The package lives under ``src/caliber`` but is not pip-installed, so ``import
caliber`` would fail with the default ``sys.path``. Adding ``src/`` here lets
every test (and any consumer running pytest from the repo root) import the
package without an editable install. Top-level ``eval`` already resolves via the
repo root that pytest inserts automatically.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
