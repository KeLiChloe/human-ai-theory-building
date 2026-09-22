from __future__ import annotations

import sys
from pathlib import Path

FORECASTS = Path(__file__).resolve().parent
LIB = FORECASTS / "_lib"
ROOT = FORECASTS.parent
for p in (ROOT, FORECASTS, LIB, ROOT / "textual_analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sorted_individuals import plot_sorted_individuals  # noqa: E402


def main() -> None:
    path = plot_sorted_individuals()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
