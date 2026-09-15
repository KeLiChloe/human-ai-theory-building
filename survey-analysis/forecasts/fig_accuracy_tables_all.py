from __future__ import annotations

import sys
from pathlib import Path

FORECASTS = Path(__file__).resolve().parent
LIB = FORECASTS / "_lib"
ROOT = FORECASTS.parent
for p in (ROOT, FORECASTS, LIB, ROOT / "textual_analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from accuracy_cosine_table import CosineEffectSource, write_cosine_me_soi_tables  # noqa: E402
import me_quant as me  # noqa: E402
import soi_quant as soi  # noqa: E402


def main() -> None:
    out = write_cosine_me_soi_tables(
        FORECASTS / "outputs",
        [
            CosineEffectSource(
                "Main Effects",
                me.group_stats,
                me.human_stats,
                me.topic_expert_stats,
                me.non_topic_expert_stats,
                me.aggregation_by_table_group,
                "sims",
                me.vecs_by_table_group,
                me.ml_vec_for_task,
            ),
            CosineEffectSource(
                "Second-Order Interactions",
                soi.stats_by_group,
                soi.stats_human,
                soi.stats_topic_expert,
                soi.stats_non_topic_expert,
                soi.aggregation_by_table_group,
                "vals",
                soi.vecs_by_table_group,
                soi.ml_vec_for_task,
            ),
        ],
        stem="fig_accuracy_tables_all",
    )
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
