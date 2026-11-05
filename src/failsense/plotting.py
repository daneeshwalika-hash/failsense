"""Colours shared by all figures (one place, so every report looks the same).

Validated categorical slots 1 and 2 (blue / orange, a colour-vision-safe pair) and
neutral inks for text, grid and reference lines. Figures are built with
``matplotlib.figure.Figure`` directly (no ``pyplot``): no global state and no GUI
backend, so they work in scripts, tests, Docker and CI.
"""

from __future__ import annotations

COLOR_TRAIN = "#2a78d6"
COLOR_VAL = "#eb6834"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
