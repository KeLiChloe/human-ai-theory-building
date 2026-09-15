from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

PACKAGE = Path(__file__).resolve().parent
LIB = PACKAGE / "_lib"
TEXTUAL = PACKAGE.parent
SURVEY = TEXTUAL.parent
for p in (SURVEY, TEXTUAL, PACKAGE, LIB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core_tail import main  # noqa: E402


if __name__ == "__main__":
    main()
