"""Pytest bootstrap: make the ``src/`` layout importable without an install step.

The package lives at ``src/caliber`` and there is no editable install, so tests
(and ad-hoc scripts) need ``src/`` on ``sys.path`` to ``import caliber``.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
