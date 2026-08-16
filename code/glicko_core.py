"""Compatibility exports for the Glicko-1 model."""

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
