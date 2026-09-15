from __future__ import annotations

import sys
from pathlib import Path

FORECASTS = Path(__file__).resolve().parent
LIB = FORECASTS / "_lib"
ROOT = FORECASTS.parent
for p in (ROOT, FORECASTS, LIB):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from equal_size_aggregation import main as equal_size_main  # noqa: E402


def main() -> None:
    equal_size_main()


if __name__ == "__main__":
    main()
