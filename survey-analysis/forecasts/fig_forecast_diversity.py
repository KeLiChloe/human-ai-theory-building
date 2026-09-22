from __future__ import annotations

import sys
from pathlib import Path

FORECASTS = Path(__file__).resolve().parent
LIB = FORECASTS / "_lib"
ROOT = FORECASTS.parent
for p in (ROOT, FORECASTS, LIB, ROOT / "textual_analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from combined_diversity_table import (  # noqa: E402
    GAIN_COEF_CSV,
    REDUNDANCY_CSV,
    write_combined_diversity_table,
)
from diversity_explains_gain import main as diversity_main  # noqa: E402
from forecast_error_redundancy import main as redundancy_main  # noqa: E402


def main() -> None:
    if not REDUNDANCY_CSV.is_file():
        print(f"Missing {REDUNDANCY_CSV.name}; building…")
        redundancy_main()
    if not GAIN_COEF_CSV.is_file():
        print(f"Missing {GAIN_COEF_CSV.name}; building…")
        diversity_main()
    path = write_combined_diversity_table()
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
