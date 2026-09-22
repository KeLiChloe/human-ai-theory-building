"""Shared visualization config for survey analysis scripts."""

# Group colors (humans + GenAI)
GROUP_COLORS = {
    "senior": "#F4A261",
    "phd": "#7DB7E8",
    "genai": "#66BB6A",
    "topic": "#B07BC9",  # soft purple, matched to other pastel group colors
    "non_topic": "#5D7A8C",
}

# Collapsed Humans (PhD + Senior Scientists); alternatives: sage #6B8E7F, teal #4A8B8B, taupe #8B7D6B
COLOR_AGG_HUMAN = "lightcoral"

# Sign-accuracy colors
SIGN_COLORS = {
    "aligned": "#66BB6A",
    "not_aligned": "#B0B0B0",
}

# ML feature / interaction descriptive bars (distinct from GenAI group green)
COLOR_ML_FEATURE_DEFAULT = "#9E9E9E"
COLOR_ML_FEATURE_HIGHLIGHT = "#002FA7"  # Klein blue; alt wine #7B2D42, brown-red #9B4D4D

