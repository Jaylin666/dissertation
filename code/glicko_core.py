"""Compatibility imports for historical scripts.

The only Glicko-1 implementation now lives in :mod:`code.models.glicko`.
New code must import that module directly.
"""

from code.models.glicko import (
    C,
    DEFAULT_RATING,
    DEFAULT_RD,
    MAX_RD,
    MIN_RD,
    Q,
    SingleGameUpdate,
    clamp_rd,
    expected_score,
    g_function,
    update_player_glicko,
    update_two_players_single_game,
)

__all__ = [
    "C",
    "DEFAULT_RATING",
    "DEFAULT_RD",
    "MAX_RD",
    "MIN_RD",
    "Q",
    "SingleGameUpdate",
    "clamp_rd",
    "expected_score",
    "g_function",
    "update_player_glicko",
    "update_two_players_single_game",
]
