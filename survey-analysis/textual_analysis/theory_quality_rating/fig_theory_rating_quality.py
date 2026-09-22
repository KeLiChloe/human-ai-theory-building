from __future__ import annotations

import os
import sys
from pathlib import Path

# Headless-safe default for CI / non-GUI runs.
os.environ.setdefault("MPLBACKEND", "Agg")

HERE = Path(__file__).resolve().parent
LIB = HERE / "_lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from quality_gap import main  # noqa: E402


if __name__ == "__main__":
    main()
