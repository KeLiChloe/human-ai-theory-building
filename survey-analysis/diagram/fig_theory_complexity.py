from __future__ import annotations

import sys
from pathlib import Path

DIAGRAM = Path(__file__).resolve().parent
LIB = DIAGRAM / "_lib"
if str(LIB) not in sys.path:
    sys.path.insert(0, str(LIB))

from analyze_diagram_complexity import main as complexity_main  # noqa: E402


def main() -> None:
    complexity_main()


if __name__ == "__main__":
    main()
