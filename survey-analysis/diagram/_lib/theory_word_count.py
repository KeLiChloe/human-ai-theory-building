"""Theory explanation word count (fourth complexity measure).

Per phase, word count is the **mean** of main-effects and SOI explanation
lengths (not their sum). Pre-ML and Post-ML enter as separate observations.

Pre-ML:  Q4 / Q10 raw theory text (user-written).
Post-ML: Q12 / Q15 LLM_refined theory text
  (descriptive panel a; phase-matched panels c–d).

Counting rule matches embedding pipeline: ``len(text.split())``.
"""

from __future__ import annotations


def word_count(text: str) -> int:
    if not text or not str(text).strip():
        return 0
    return len(str(text).split())


def theory_text_column(task: str, *, phase: str = "Pre") -> str:
    """CSV column for main-effects theory text (Race / Gender)."""
    return theory_text_columns(task, phase=phase)[0]


def theory_text_columns(task: str, *, phase: str = "Pre") -> tuple[str, str]:
    """``(main_effects, SOI)`` theory text columns for ``task`` × ``phase``."""
    if phase in ("Pre", "Pre-ML"):
        return (
            f"Q {task}.4 pre-ML theory (main effects)",
            f"Q {task}.10 pre-ML theory (SOI)",
        )
    if phase in ("Post", "Post-ML"):
        return (
            f"Q {task}.12 LLM_refined post-ML theory (main effects)",
            f"Q {task}.15 LLM_refined post-ML theory (SOI)",
        )
    raise ValueError(f"Unknown phase: {phase!r}")


def combined_theory_word_count(me_text: str, soi_text: str) -> float:
    """Mean of main-effects and SOI theory word counts (non-empty texts only).

    If only one of ME/SOI is non-empty, returns that length. If both empty,
    returns 0.
    """
    parts = [word_count(me_text), word_count(soi_text)]
    parts = [w for w in parts if w > 0]
    if not parts:
        return 0.0
    return float(sum(parts) / len(parts))
