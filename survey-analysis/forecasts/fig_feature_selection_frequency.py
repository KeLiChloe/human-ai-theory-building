from __future__ import annotations

import sys
from pathlib import Path

FORECASTS = Path(__file__).resolve().parent
LIB = FORECASTS / "_lib"
ROOT = FORECASTS.parent
for p in (ROOT, FORECASTS, LIB, ROOT / "textual_analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from feature_selection_frequency import plot_feature_selection_frequency  # noqa: E402


def main() -> None:
    out = FORECASTS / "outputs" / "fig_feature_selection_frequency"
    path = plot_feature_selection_frequency(out)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
